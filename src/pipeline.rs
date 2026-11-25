//! Port of app/translator/pipeline.py.
//!
//! Orchestrates batching and LLM calls, tracks progress, and supports
//! cancellation. Progress updates happen under a short std::sync::Mutex lock
//! (no .await while holding it).

use crate::config::Settings;
use crate::db;
use crate::glossary::GlossaryEntry;
use crate::llm::LlmEngine;
use crate::subtitle::SubtitleFile;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Per-job progress, mirrored over /api/progress polling.
#[derive(Clone, Debug)]
pub struct TranslationProgress {
    pub total_batches: usize,
    pub completed_batches: usize,
    pub status: String,
    pub error: Option<String>,
    pub last_batch_seconds: f64,
    pub device_used: String,
    pub completed_lines: usize,
    pub total_files: usize,
    pub completed_files: usize,
    pub current_file: String,
}

impl Default for TranslationProgress {
    fn default() -> Self {
        Self {
            total_batches: 0,
            completed_batches: 0,
            status: "idle".to_string(),
            error: None,
            last_batch_seconds: 0.0,
            device_used: "local-llm".to_string(),
            completed_lines: 0,
            total_files: 1,
            completed_files: 0,
            current_file: String::new(),
        }
    }
}

impl TranslationProgress {
    pub fn percent(&self) -> f64 {
        if self.total_batches == 0 {
            return 0.0;
        }
        // Clamp: with dedupe/translation-memory the real batch count can be
        // smaller than the upfront estimate, so completed may pass the
        // estimate mid-job.
        ((self.completed_batches as f64 / self.total_batches as f64) * 100.0).min(100.0)
    }
}

/// Short-lived lock helper: recover from a poisoned lock rather than panicking
/// (a poisoned lock would mean a panicking writer; the state is still usable).
fn lock(
    progress: &Arc<Mutex<TranslationProgress>>,
) -> MutexGuard<'_, TranslationProgress> {
    progress.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn round2(secs: f64) -> f64 {
    (secs * 100.0).round() / 100.0
}

pub struct Pipeline {
    pub llm: LlmEngine,
}

/// Result of one file translation: texts aligned to `sub.lines` plus the
/// number of LLM batches actually run (dedupe/TM can shrink it below the
/// initial estimate).
pub struct TranslateOutput {
    pub texts: Vec<String>,
    pub batches_done: usize,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            llm: LlmEngine::new(),
        }
    }

    /// Translate a subtitle file, returning translations aligned to
    /// `sub.lines` (same length and order as the original line texts).
    ///
    /// Efficiency passes, in order:
    /// 1. Dedupe — identical (trimmed) lines are translated once and mapped
    ///    back to every position that uses them.
    /// 2. Translation memory — SQLite cache keyed by (source, target, text);
    ///    hits skip the LLM entirely, misses are batched and stored back.
    /// 3. Cross-batch context — each batch prompt carries the preceding
    ///    dialogue lines so continuity survives batch boundaries.
    pub async fn translate(
        &self,
        sub: &SubtitleFile,
        source_lang: &str,
        target_lang: &str,
        glossary: Option<&[GlossaryEntry]>,
        batch_size: usize,
        temperature: f64,
        max_tokens: u32,
        top_p: f64,
        repair: bool,
        progress: Arc<Mutex<TranslationProgress>>,
        cancel: Arc<AtomicBool>,
    ) -> TranslateOutput {
        let cfg = Settings::global();
        let concurrency = cfg.llm_concurrency.max(1);

        let lines = &sub.lines;
        if lines.is_empty() {
            lock(&progress).status = "completed".to_string();
            return TranslateOutput { texts: Vec::new(), batches_done: 0 };
        }

        let original_texts: Vec<String> = lines.iter().map(|l| l.text.clone()).collect();

        // ------------------------------------------------------------------
        // 0. Context analysis pass (one cheap LLM call per file): extracts
        // characters/terms/tone from the opening lines into a brief that is
        // injected into the system prompt for every batch of this file.
        // ------------------------------------------------------------------
        let mut context_brief: Option<String> = None;
        if cfg.analyze_enabled && original_texts.len() >= 10 {
            lock(&progress).status = "analyzing".to_string();
            context_brief = self.llm.analyze_context(&original_texts, cfg.analyze_lines).await;
            if cancel.load(Ordering::SeqCst) {
                lock(&progress).status = "cancelled".to_string();
                return TranslateOutput { texts: Vec::new(), batches_done: 0 };
            }
        }

        lock(&progress).status = "translating".to_string();

        // ------------------------------------------------------------------
        // 1. Dedupe unique non-empty trimmed texts (first-appearance order).
        // ------------------------------------------------------------------
        let mut uniq_map: HashMap<String, usize> = HashMap::new();
        let mut uniq_texts: Vec<String> = Vec::new();
        let mut uniq_first_orig: Vec<usize> = Vec::new();
        // Per line: slot in uniq_texts, or None for empty lines (never sent).
        let mut line_slot: Vec<Option<usize>> = vec![None; original_texts.len()];
        for (li, t) in original_texts.iter().enumerate() {
            let trimmed = t.trim();
            if trimmed.is_empty() {
                continue;
            }
            let next = uniq_texts.len();
            let slot = *uniq_map.entry(trimmed.to_owned()).or_insert(next);
            if slot == next {
                uniq_texts.push(trimmed.to_owned());
                uniq_first_orig.push(li);
            }
            line_slot[li] = Some(slot);
        }

        // ------------------------------------------------------------------
        // 2. Translation memory lookups (blocking SQLite, off the runtime).
        // ------------------------------------------------------------------
        let mut uniq_out: Vec<Option<String>> = vec![None; uniq_texts.len()];
        if cfg.tm_enabled && !uniq_texts.is_empty() {
            let texts = uniq_texts.clone();
            let src = source_lang.to_string();
            let tgt = target_lang.to_string();
            match tokio::task::spawn_blocking(move || db::tm_lookup(&texts, &src, &tgt)).await {
                Ok(Ok(hits)) => {
                    let hit_count = hits.iter().filter(|h| h.is_some()).count();
                    if hit_count > 0 {
                        tracing::info!(
                            "Translation memory: {}/{} line(s) reused from cache",
                            hit_count,
                            uniq_texts.len()
                        );
                    }
                    uniq_out = hits;
                }
                Ok(Err(e)) => tracing::warn!("Translation memory lookup failed: {}", e),
                Err(e) => tracing::warn!("Translation memory lookup join error: {}", e),
            }
        }

        // Remaining unique texts that still need the LLM, as slots.
        let misses: Vec<usize> = (0..uniq_texts.len())
            .filter(|&i| uniq_out[i].is_none())
            .collect();
        let size = batch_size.max(1);
        let miss_batches: Vec<Vec<usize>> =
            misses.chunks(size).map(|c| c.to_vec()).collect();
        lock(&progress).total_batches = miss_batches.len();

        // Per-batch work items: texts to send plus preceding-dialogue context
        // (window in ORIGINAL file order right before the earliest line of
        // the batch, so continuity matches the real timeline).
        struct BatchWork {
            texts: Vec<String>,
            prev_context: Vec<String>,
        }
        let ctx_lines = cfg.context_lines.min(10);
        let work: Vec<BatchWork> = miss_batches
            .iter()
            .map(|slots| {
                let first_idx = slots
                    .iter()
                    .map(|&u| uniq_first_orig[u])
                    .min()
                    .unwrap_or(0);
                let ctx_start = first_idx.saturating_sub(ctx_lines);
                let prev_context: Vec<String> = original_texts[ctx_start..first_idx]
                    .iter()
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect();
                BatchWork {
                    texts: slots.iter().map(|&u| uniq_texts[u].clone()).collect(),
                    prev_context,
                }
            })
            .collect();

        let mut all_uniq_results: Vec<Vec<String>> = vec![Vec::new(); work.len()];

        // Sequential mode
        if concurrency <= 1 {
            for (bi, item) in work.iter().enumerate() {
                if cancel.load(Ordering::SeqCst) {
                    lock(&progress).status = "cancelled".to_string();
                    return TranslateOutput {
                        texts: Vec::new(),
                        batches_done: bi,
                    };
                }

                let batch_start = std::time::Instant::now();
                let translated = self
                    .llm
                    .translate_batch(
                        &item.texts,
                        source_lang,
                        target_lang,
                        glossary,
                        context_brief.as_deref(),
                        temperature,
                        max_tokens,
                        top_p,
                        repair,
                        &item.prev_context,
                    )
                    .await;
                all_uniq_results[bi] = translated;

                let batch_elapsed = batch_start.elapsed().as_secs_f64();
                {
                    let mut p = lock(&progress);
                    p.completed_batches += 1;
                    p.completed_lines = all_uniq_results[..=bi]
                        .iter()
                        .map(|v| v.len())
                        .sum::<usize>();
                    p.last_batch_seconds = round2(batch_elapsed);
                    p.device_used = "local-llm".to_string();
                }
            }
        } else {
            // Concurrent mode. Results are keyed by batch_index so ordering is
            // preserved; each future owns its data and borrows only the engine.
            let mut results_by_index: HashMap<usize, Vec<String>> = HashMap::new();

            let phase_start = std::time::Instant::now();

            let futs: Vec<_> = work
                .iter()
                .enumerate()
                .map(|(bi, item)| {
                    let texts = item.texts.clone();
                    let prev_context = item.prev_context.clone();
                    let brief = context_brief.clone();
                    let engine = &self.llm;
                    let cancel_f = Arc::clone(&cancel);
                    async move {
                        if cancel_f.load(Ordering::SeqCst) {
                            // Cancelled before starting: keep alignment by
                            // returning originals (caller discards them).
                            (bi, texts)
                        } else {
                            let result = engine
                                .translate_batch(
                                    &texts,
                                    source_lang,
                                    target_lang,
                                    glossary,
                                    brief.as_deref(),
                                    temperature,
                                    max_tokens,
                                    top_p,
                                    repair,
                                    &prev_context,
                                )
                                .await;
                            (bi, result)
                        }
                    }
                })
                .collect();

            let stream = futures::stream::iter(futs).buffer_unordered(concurrency);

            let mut stream = Box::pin(stream);
            while let Some((bi, translated)) = stream.next().await {
                {
                    let mut p = lock(&progress);
                    p.completed_batches += 1;
                    p.completed_lines = results_by_index.values().map(|v| v.len()).sum();
                    p.last_batch_seconds = round2(phase_start.elapsed().as_secs_f64());
                    p.device_used = "local-llm".to_string();
                }
                results_by_index.insert(bi, translated);
            }

            // Stitch results back in batch order.
            for (bi, _) in work.iter().enumerate() {
                if let Some(translated) = results_by_index.get(&bi) {
                    all_uniq_results[bi] = translated.clone();
                }
            }
        }

        if cancel.load(Ordering::SeqCst) {
            lock(&progress).status = "cancelled".to_string();
            return TranslateOutput { texts: Vec::new(), batches_done: work.len() };
        }

        // ------------------------------------------------------------------
        // Map unique results back to per-line outputs.
        // ------------------------------------------------------------------
        for (bi, slots) in miss_batches.iter().enumerate() {
            for (pos, &u) in slots.iter().enumerate() {
                match all_uniq_results.get(bi).and_then(|v| v.get(pos)) {
                    Some(t) if !t.trim().is_empty() => uniq_out[u] = Some(t.clone()),
                    _ => {} // stays None -> original kept below
                }
            }
        }

        let mut all_translated: Vec<String> = vec![String::new(); original_texts.len()];
        for (li, slot) in line_slot.iter().enumerate() {
            match slot {
                Some(u) => match &uniq_out[*u] {
                    Some(t) if !t.trim().is_empty() => all_translated[li] = t.clone(),
                    // Untranslated -> keep original, preserves alignment.
                    _ => all_translated[li] = original_texts[li].clone(),
                },
                // Empty/whitespace line: never sent to the LLM.
                None => all_translated[li] = String::new(),
            }
        }

        // ------------------------------------------------------------------
        // Store new translations in the translation memory. Echoes (result ==
        // original) are NOT stored — they are more likely a failed request
        // than a legitimate identity translation.
        // ------------------------------------------------------------------
        if cfg.tm_enabled {
            let save_entries: Vec<(String, String)> = uniq_texts
                .iter()
                .zip(uniq_out.iter())
                .filter_map(|(s, t)| {
                    t.as_ref()
                        .filter(|t| !t.trim().is_empty() && !t.trim().eq_ignore_ascii_case(s.trim()))
                        .map(|t| (s.clone(), t.clone()))
                })
                .collect();
            if !save_entries.is_empty() {
                let src = source_lang.to_string();
                let tgt = target_lang.to_string();
                match tokio::task::spawn_blocking(move || db::tm_save(&save_entries, &src, &tgt))
                    .await
                {
                    Ok(Ok(n)) if n > 0 => {
                        tracing::info!("Translation memory: stored {} new entr(y|ies)", n)
                    }
                    Ok(Err(e)) => tracing::warn!("Translation memory save failed: {}", e),
                    Err(e) => tracing::warn!("Translation memory save join error: {}", e),
                    _ => {}
                }
            }
        }

        // Validate: detect lines that weren't translated
        let untranslated = original_texts
            .iter()
            .zip(all_translated.iter())
            .filter(|(orig, trans)| !orig.trim().is_empty() && !orig.trim().eq_ignore_ascii_case(trans.trim()))
            .count();
        if untranslated > 0 {
            tracing::warn!(
                "{}/{} lines may not have been translated (still match original text)",
                untranslated,
                original_texts.len()
            );
        }

        lock(&progress).completed_lines = all_translated.len();
        lock(&progress).status = "completed".to_string();

        TranslateOutput {
            texts: all_translated,
            batches_done: work.len(),
        }
    }
}
