//! Port of app/translator/llm_engine.py.
//!
//! Translation engine using a local LLM via llama-server (OpenAI-compatible
//! API). The engine is a stateless HTTP client — no model weights are loaded
//! here. One reusable reqwest::Client (internally Arc'd, thread-safe) is
//! shared across concurrent batches to keep the connection pool warm.

use crate::config::Settings;
use crate::glossary::GlossaryEntry;
use crate::resolver;
use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Compact translation system prompt — every token counts at generation speed.
/// Byte-exact copy of BASE_SYSTEM_PROMPT in llm_engine.py (incl. trailing blank line).
const BASE_SYSTEM_PROMPT: &str = "You are a professional subtitle translator. ALWAYS translate every line from source to target language. NEVER keep the original text.\n\nExample of CORRECT output (from Indonesian to English):\nInput:  1. Halo, apa kabar?\n        2. Saya baikah, terima kasih.\nOutput: 1. Hello, how are you?\n        2. I'm fine, thank you.\n\nRules:\n- Return numbered translations matching the input numbering exactly.\n- Each line: <number>. <translation>\n- Output EXACTLY one numbered line per input line — never skip, merge, split, or add lines.\n- Keep concise and natural for subtitle display.\n- Preserve meaning, tone, and register; translate idioms naturally, not word-by-word.\n- Keep proper nouns and character names unchanged; translate ALL common nouns and loanwords into the target language.\n- Do NOT add explanations, notes, or any text outside the numbered lines.\n- Never repeat the original text — always produce the translated version.\n- Even one word lines must be translated (e.g., \"Oke\" -> \"Okay\").\n\n";

/// System prompt for the one-shot context analysis pass. Strict: the model
/// must return only the brief, no preamble, no translation.
const ANALYSIS_SYSTEM_PROMPT: &str = "You analyze the opening subtitle lines of a video to help a subtitle translator. Output ONLY a compact brief, at most 8 short bullet lines:\n- Characters: personal names exactly as they appear\n- Places/organizations: proper names only, exactly as written\n- Tone: one short phrase (e.g. casual, formal, comedic)\nDo NOT list common nouns or ordinary objects. Do NOT translate anything. No preamble, no explanations.\n\n";

/// Result of probing the LLM server (`/v1/models`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct LlmStatus {
    pub connected: bool,
    pub model: Option<String>,
    pub base_url: String,
    pub error: Option<String>,
}

/// Matches a numbered output line: "1. text", "2) text", "3: text".
/// The separator must be followed by whitespace or end-of-line, and the digit
/// run is capped, so SRT-style timelines ("00:00:01,000 --> ...") and decimals
/// ("3.5") are NOT treated as numbered translation lines.
/// Byte-parity with _NUMBERED_RE in llm_engine.py.
fn numbered_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*(\d{1,4})\s*[.):](?:\s+(.*))?$").unwrap())
}

/// Convert a JSON list to a list of stripped strings (port of _coerce_strings:
/// only str, int, float elements are kept).
fn coerce_strings(data: &[Value]) -> Vec<String> {
    data.iter()
        .filter_map(|x| match x {
            Value::String(s) => Some(s.trim().to_string()),
            Value::Number(n) => Some(n.to_string().trim().to_string()),
            _ => None,
        })
        .collect()
}

/// Build the system prompt for translation, including glossary if provided.
/// Port of build_system_prompt(). Glossary entries are sorted deterministically
/// so the prompt is byte-stable across batches (llama-server prompt-prefix cache).
/// `context_brief` (optional output of the analysis pass) is appended last;
/// it is constant for a whole file, so per-file prompt caching still works.
pub fn build_system_prompt(
    source_lang: &str,
    target_lang: &str,
    glossary: Option<&[GlossaryEntry]>,
    context_brief: Option<&str>,
) -> String {
    let mut prompt = format!("{}\nTranslate from {} to {}.", BASE_SYSTEM_PROMPT, source_lang, target_lang);

    if let Some(glossary) = glossary.filter(|g| !g.is_empty()) {
        let mut entries: Vec<&GlossaryEntry> = glossary.iter().collect();
        entries.sort_by(|a, b| (&a.source, &a.target).cmp(&(&b.source, &b.target)));
        let terms: Vec<String> = entries
            .iter()
            .map(|e| format!("- {} => {}", e.source, e.target))
            .collect();
        if !terms.is_empty() {
            prompt.push_str(
                "\n\nUse these glossary terms when translating (ALWAYS use the given translation, preserving capitalization where appropriate):\n",
            );
            prompt.push_str(&terms.join("\n"));
        }
    }

    if let Some(brief) = context_brief.map(str::trim).filter(|b| !b.is_empty()) {
        prompt.push_str(
            "\n\nCONTEXT BRIEF for this video (use it for consistent character names, terms, and tone):\n",
        );
        prompt.push_str(brief);
    }

    prompt
}

pub struct LlmEngine {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl LlmEngine {
    pub fn new() -> Self {
        let cfg = Settings::global();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.llm_timeout_secs))
            .connect_timeout(Duration::from_secs(cfg.llm_connect_timeout_secs))
            .pool_max_idle_per_host(std::cmp::max(4, cfg.llm_concurrency * 2))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url: cfg.llm_base_url.clone(),
            model: cfg.llm_model_name.clone(),
        }
    }

    /// Check if llama-server is reachable and get model info.
    pub async fn check_connection(&self) -> LlmStatus {
        let base_url = self.base_url.clone();
        let url = format!("{}/v1/models", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if resp.status() != reqwest::StatusCode::OK {
                    // Python parity: any non-200 falls to the "no models" branch.
                    return LlmStatus {
                        connected: false,
                        model: None,
                        base_url,
                        error: Some("No models loaded on server".to_string()),
                    };
                }
                match resp.json::<Value>().await {
                    Ok(data) => {
                        let models_empty = data
                            .get("data")
                            .and_then(|v| v.as_array())
                            .map(|a| a.is_empty())
                            .unwrap_or(true);
                        if !models_empty {
                            let models = data.get("data").and_then(|v| v.as_array()).unwrap();
                            // Python: models[0].get("id", "unknown")
                            let id = models[0]
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            LlmStatus {
                                connected: true,
                                model: Some(id),
                                base_url,
                                error: None,
                            }
                        } else {
                            LlmStatus {
                                connected: false,
                                model: None,
                                base_url,
                                error: Some("No models loaded on server".to_string()),
                            }
                        }
                    }
                    Err(e) => LlmStatus {
                        connected: false,
                        model: None,
                        base_url,
                        error: Some(e.to_string()),
                    },
                }
            }
            Err(e) => LlmStatus {
                connected: false,
                model: None,
                base_url,
                error: Some(e.to_string()),
            },
        }
    }

    /// Translate a batch of subtitle lines using the LLM.
    ///
    /// Sends all lines in a single numbered prompt so the LLM has full context
    /// for accurate, context-aware translation. On transport errors or
    /// malformed responses the batch degrades gracefully and returns the
    /// original texts. A second "repair pass" retranslates any lines the first
    /// response left out (missing numbers, truncation fragments) before
    /// giving up on them.
    ///
    /// `prev_context` holds the immediately preceding dialogue lines (already
    /// in the source language, unnumbered) for cross-batch continuity.
    #[allow(clippy::too_many_arguments)]
    pub async fn translate_batch(
        &self,
        texts: &[String],
        source_lang: &str,
        target_lang: &str,
        glossary: Option<&[GlossaryEntry]>,
        context_brief: Option<&str>,
        temperature: f64,
        max_tokens: u32,
        top_p: f64,
        repair: bool,
        prev_context: &[String],
    ) -> Vec<String> {
        if texts.is_empty() {
            return Vec::new();
        }
        let cfg = Settings::global();

        // Filter out empty lines but track their positions
        let valid_indices: Vec<usize> = (0..texts.len())
            .filter(|&j| !texts[j].trim().is_empty())
            .collect();
        if valid_indices.is_empty() {
            return vec![String::new(); texts.len()];
        }

        let valid_texts: Vec<&String> = valid_indices.iter().map(|&j| &texts[j]).collect();

        // Build numbered input
        let numbered_input: String = valid_texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{}. {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n");

        // Inject pronoun resolution context (e.g., "Dia" -> "He"/"She").
        // Kept out of the system prompt so the system prompt stays cache-stable.
        let mut context = String::new();
        if valid_texts.len() >= 3 {
            let owned: Vec<String> = valid_texts.iter().map(|t| (*t).clone()).collect();
            context.push_str(&resolver::build_prompt_hint(&owned));
        }
        // Cross-batch dialogue context. Unnumbered ("- " bullets) so the
        // numbered-response parser can never confuse it with output lines.
        if !prev_context.is_empty() {
            let lines: Vec<String> = prev_context
                .iter()
                .map(|t| format!("- {}", t.trim()))
                .collect();
            context.push_str(&format!(
                "\n\nPREVIOUS DIALOGUE (for context only — do NOT translate, do NOT include in output):\n{}",
                lines.join("\n")
            ));
        }

        // Stating the expected line count up front measurably reduces dropped
        // lines with local models.
        let user_prompt = format!(
            "Translate these {} subtitle lines:\n\n{}{}",
            valid_texts.len(),
            numbered_input,
            context
        );

        let build_payload = |user: &str, with_brief: bool, temp: f64| {
            let system = build_system_prompt(
                source_lang,
                target_lang,
                glossary,
                if with_brief { context_brief } else { None },
            );
            serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temp,
                "max_tokens": max_tokens,
                "top_p": top_p,
                // Qwen3.x default to thinking mode which consumes the whole
                // token budget on reasoning_content (content stays empty).
                "reasoning_effort": "none",
            })
        };

        let t0 = Instant::now();

        let attempts = std::cmp::max(1, cfg.llm_max_retries as usize + 1);
        let mut content: Option<String> = None;
        let mut finish_reason = String::new();
        let mut last_error = "unknown error".to_string();

        let payload = build_payload(&user_prompt, true, temperature);
        for attempt in 0..attempts {
            // Small exponential backoff so a busy llama-server is not hammered
            // back-to-back (retry 1: 250ms, retry 2: 500ms, ...).
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(250 * attempt as u64)).await;
            }
            match self.chat_completion(&payload).await {
                Ok((c, f)) => {
                    content = Some(c);
                    finish_reason = f;
                    break;
                }
                Err(e) => {
                    last_error = e;
                    content = None;
                    tracing::warn!(
                        "LLM batch attempt {}/{} failed ({}), retrying",
                        attempt + 1,
                        attempts,
                        last_error
                    );
                }
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();

        let content = match content {
            Some(c) => c,
            None => {
                tracing::error!(
                    "LLM request failed after {} attempt(s): {}",
                    attempts,
                    last_error
                );
                return texts.to_vec(); // Return originals on failure
            }
        };

        // Parse translations (numbered primary, JSON fallback)
        let mut translated = parse_response(&content, valid_texts.len());
        tracing::debug!(
            "First response ({} chars): {}",
            content.len(),
            content.chars().take(300).collect::<String>()
        );

        // Detect truncation: if the LLM hit the token ceiling, the last
        // line may be a partial fragment. Blank it so the repair pass
        // retranslates it cleanly. (Without repair the blank maps back to
        // the original text, same graceful degradation as before.)
        if finish_reason == "length" && !translated.is_empty() {
            let last = translated.last().unwrap().trim().to_string();
            let word_count = last.split_whitespace().count();
            let char_count = last.chars().count();
            if word_count < 2 && char_count < 20 {
                tracing::warn!("Last translation line looks truncated, scheduling repair");
                *translated.last_mut().unwrap() = String::new();
            }
        }

        // Repair is opt-in (UI checkbox, default off): it adds retry rounds
        // that push coverage toward 100% at the cost of extra LLM requests.
        // With repair off, unfixed lines keep the original text — parity
        // with the plain Python engine's speed.
        if repair {
        // Repair loop: retranslates entries the first response left empty
        // (missing numbers, truncated fragments) AND entries that merely
        // echoed the original text (a common local-model failure that would
        // otherwise sail through parsing as a "successful" translation).
        //
        // Small local models sometimes echo a whole batch even when asked
        // again, so repair is iterative: rounds 1-2 re-request the missing
        // lines as a numbered batch; round 3 falls back to one request per
        // stubborn line (single-line prompts almost never echo). Each round
        // only pays for the lines that are still unfixed; unfixed lines keep
        // the original text so alignment never drifts.
        let mut round = 0usize;
        // Set when a batch-style repair round recovers nothing: the model
        // echoed the repair request too, so another batch retry is wasted —
        // the next round switches to per-line requests immediately.
        let mut batch_repair_dead = false;
        let mut perline_passes = 0usize;
        loop {
            let missing: Vec<usize> = (0..translated.len())
                .filter(|&i| {
                    let t = translated[i].trim();
                    t.is_empty() || t.eq_ignore_ascii_case(valid_texts[i].trim())
                })
                .collect();
            if missing.is_empty() {
                break;
            }
            round += 1;
            let mut recovered = 0usize;

            if round <= 2 && !batch_repair_dead {
                // Numbered multi-line repair round.
                let repair_texts: Vec<&String> =
                    missing.iter().map(|&i| valid_texts[i]).collect();
                let repair_input: String = repair_texts
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("{}. {}", i + 1, t))
                    .collect::<Vec<_>>()
                    .join("\n");
                let repair_user = format!(
                    "These lines were NOT translated — the first attempt returned the original text or nothing. Translate each line from {} to {} properly. Do NOT repeat the input text:\n\n{}",
                    source_lang,
                    target_lang,
                    repair_input
                );
                let repair_payload = build_payload(&repair_user, false, temperature);
                match self.chat_completion(&repair_payload).await {
                    Ok((repair_content, _)) => {
                        tracing::debug!(
                            "Repair round {} response ({} chars): {}",
                            round,
                            repair_content.len(),
                            repair_content.chars().take(300).collect::<String>()
                        );
                        let repaired = parse_response(&repair_content, missing.len());
                        for ((pos, &slot), txt) in
                            missing.iter().enumerate().zip(repaired.iter())
                        {
                            // A line counts as recovered only when it is
                            // non-empty AND no longer an echo of the input.
                            if !txt.trim().is_empty()
                                && !txt.trim().eq_ignore_ascii_case(repair_texts[pos].trim())
                            {
                                translated[slot] = txt.clone();
                                recovered += 1;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Repair round {} failed ({}), keeping originals",
                            round,
                            e
                        );
                    }
                }
                if recovered == 0 {
                    batch_repair_dead = true;
                }
            } else {
                // Final rounds: one small request per stubborn line, at most
                // two passes. The second pass retries at a higher sampling
                // temperature, which breaks echo loops the normal temperature
                // keeps falling into (deterministic echo of the input). A
                // third pass would repeat the identical prompt pointlessly.
                perline_passes += 1;
                let retry_temp = if perline_passes >= 2 { 0.8 } else { 0.2 };
                for &slot in &missing {
                    let text = valid_texts[slot];
                    let user = format!(
                        "Translate this subtitle line from {} to {}. Output ONLY the translation, nothing else. Do NOT repeat the input text:\n\n{}",
                        source_lang, target_lang, text
                    );
                    let payload = build_payload(&user, false, retry_temp);
                    match self.chat_completion(&payload).await {
                        Ok((content, _)) => {
                            let mut line = content
                                .lines()
                                .map(str::trim)
                                .find(|l| !l.is_empty())
                                .unwrap_or_default()
                                .trim_start_matches(['-', '*', ' '])
                                .trim()
                                .trim_matches('"')
                                .to_string();
                            // Models often number even single-line answers
                            // ("1. From my...") despite the instruction;
                            // strip a leading numbering prefix.
                            if let Some(caps) = numbered_re().captures(&line) {
                                if let Some(m) = caps.get(2) {
                                    let t = m.as_str().trim().to_string();
                                    if !t.is_empty() {
                                        line = t;
                                    }
                                }
                            }
                            if !line.is_empty() && !line.eq_ignore_ascii_case(text.trim()) {
                                translated[slot] = line;
                                recovered += 1;
                            }
                        }
                        Err(_) => continue,
                    }
                }
                let total = missing.len();
                tracing::info!(
                    "Repair round {} (per-line, temp {retry_temp}) recovered {recovered}/{total} missing line(s)",
                    round
                );
                if perline_passes >= 2 {
                    break;
                }
            }
            tracing::info!(
                "Repair round {} recovered {}/{} missing line(s)",
                round,
                recovered,
                missing.len()
            );
        } // end repair loop
        } // if repair (opt-in)

        // Map back to full list. Empty/whitespace entries keep the original
        // text for that specific line so alignment never drifts.
        let mut results: Vec<String> = vec![String::new(); texts.len()];
        for (&idx, text) in valid_indices.iter().zip(translated.iter()) {
            if text.trim().is_empty() {
                results[idx] = texts[idx].clone(); // untranslated -> keep original, preserves alignment
            } else {
                results[idx] = text.clone();
            }
        }

        tracing::info!(
            "Translated {} lines via LLM in {:.3}s",
            valid_texts.len(),
            elapsed
        );
        results
    }

    /// One-shot context analysis pass (best effort). Reads up to `max_lines`
    /// opening dialogue lines and returns a compact brief (characters, terms,
    /// tone) to inject into the translation system prompt. Returns None on any
    /// failure — translation then proceeds without the brief.
    ///
    /// One extra LLM call per file; output is capped so the system prompt
    /// stays small. The subtitle text is untrusted input: the strict analysis
    /// system prompt keeps the output bound to a short bullet brief, and the
    /// brief itself is length-capped here before it ever reaches a prompt.
    pub async fn analyze_context(&self, lines: &[String], max_lines: usize) -> Option<String> {
        if lines.is_empty() || max_lines == 0 {
            return None;
        }
        let sample: Vec<&str> = lines
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .take(max_lines)
            .collect();
        if sample.len() < 10 {
            // Too short to be worth an extra call.
            return None;
        }
        let user_prompt = sample.join("\n");
        let payload = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 256,
            "reasoning_effort": "none",
        });
        let t0 = Instant::now();
        match self.chat_completion(&payload).await {
            Ok((content, _)) => {
                let joined: String = content
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .take(10)
                    .collect::<Vec<_>>()
                    .join("\n");
                // Safe char-boundary truncation (String::truncate panics on
                // multibyte boundaries).
                let brief = match joined.char_indices().nth(600) {
                    Some((idx, _)) => joined[..idx].to_string(),
                    None => joined,
                };
                tracing::info!("Context brief generated in {:.2}s", t0.elapsed().as_secs_f64());
                tracing::debug!("Context brief:\n{}", brief);
                Some(brief)
            }
            Err(e) => {
                tracing::warn!("Context analysis failed ({}); translating without brief", e);
                None
            }
        }
    }

    /// Single chat-completion request. Returns (content, finish_reason).
    /// The payload is passed in so retries reuse it without re-serializing.
    async fn chat_completion(&self, payload: &Value) -> Result<(String, String), String> {
        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(payload)
            .send()
            .await
            .map_err(|e| format!("transport error: {e}"))?;
        let resp = resp
            .error_for_status()
            .map_err(|e| format!("http status error: {e}"))?;
        let result: Value = resp
            .json()
            .await
            .map_err(|e| format!("json decode error: {e}"))?;
        let choice = result
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|c| c.first())
            .ok_or_else(|| "malformed response (missing choices)".to_string())?;
        let message = choice
            .get("message")
            .ok_or_else(|| "malformed response (missing message)".to_string())?;
        let content = match message.get("content") {
            Some(Value::String(s)) => s.trim().to_string(),
            Some(Value::Null) | None => String::new(),
            Some(_) => {
                return Err("malformed response (content not a string)".to_string());
            }
        };
        let finish = choice
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok((content, finish))
    }
}

/// Pad with "" or truncate to exactly `expected_count` entries.
fn pad_truncate(mut parsed: Vec<String>, expected_count: usize) -> Vec<String> {
    if parsed.len() < expected_count {
        parsed.resize(expected_count, String::new());
    } else if parsed.len() > expected_count {
        parsed.truncate(expected_count);
    }
    parsed
}

/// Port of _parse_response. Tries numbered lines first (primary format),
/// then JSON array (legacy/fallback), then raw-line fallback.
/// Always returns a list of length `expected_count` (padded with "").
fn parse_response(content: &str, expected_count: usize) -> Vec<String> {
    // 1. Numbered lines (primary — matches prompt format)
    let parsed = try_numbered(content);
    if !parsed.is_empty() {
        return pad_truncate(parsed, expected_count);
    }

    // 2. JSON array (Qwen sometimes returns this anyway)
    if let Some(parsed) = try_json_array(content) {
        if !parsed.is_empty() {
            return pad_truncate(parsed, expected_count);
        }
    }

    // 3. Last-resort raw line extraction
    let parsed = line_fallback(content, expected_count);
    pad_truncate(parsed, expected_count)
}

/// Port of _try_numbered. Parses by explicit number so empty entries do not
/// shift every later line. Returns a vector covering the numbering range, with
/// "" for any missing/empty number. Empty vector if no numbered lines are
/// present. Absurd numbers are ignored to avoid huge allocations.
fn try_numbered(content: &str) -> Vec<String> {
    let re = numbered_re();
    let mut entries: std::collections::BTreeMap<usize, String> = std::collections::BTreeMap::new();
    for line in content.split('\n') {
        if let Some(caps) = re.captures(line) {
            // \d{1,4} always parses to usize here
            let n: usize = caps.get(1).unwrap().as_str().parse().unwrap_or(usize::MAX);
            if n > 10000 {
                // ignore absurd numbers (hallucinated timestamps etc.)
                continue;
            }
            let text = caps
                .get(2)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            entries.insert(n, text);
        }
    }
    if entries.is_empty() {
        return Vec::new();
    }
    // Detect 0-based numbering so line 0 is not silently dropped.
    let base: usize = if entries.contains_key(&0) { 0 } else { 1 };
    let max_num = *entries.keys().next_back().unwrap();
    let size = max_num - base + 1;
    let mut out = vec![String::new(); size];
    for (&n, txt) in &entries {
        let idx = n - base;
        if idx < size {
            out[idx] = txt.clone();
        }
    }
    out
}

/// Port of _try_json_array. Extract a JSON array of strings from the response.
/// Handles strict JSON plus Qwen-style arrays that omit commas between string
/// elements (e.g. ["a" "b" "c"]).
fn try_json_array(content: &str) -> Option<Vec<String>> {
    static FENCE_RE: OnceLock<Regex> = OnceLock::new();
    static ARRAY_RE: OnceLock<Regex> = OnceLock::new();
    static STRING_RE: OnceLock<Regex> = OnceLock::new();
    let fence_re = FENCE_RE.get_or_init(|| Regex::new(r"```(?:json)?\s*").unwrap());
    let array_re = ARRAY_RE.get_or_init(|| Regex::new(r"\[[\s\S]*\]").unwrap());
    let string_re = STRING_RE.get_or_init(|| Regex::new(r#""((?:[^"\\]|\\.)*)""#).unwrap());

    // Strip markdown code fences
    let cleaned = fence_re.replace_all(content, "").trim().to_string();
    // Find the bracketed array region (first '[' through last ']')
    let array_text = array_re.find(&cleaned)?.as_str();

    // 1. Strict JSON parse
    if let Ok(data) = serde_json::from_str::<Value>(array_text) {
        if let Value::Array(items) = data {
            return Some(coerce_strings(&items));
        }
    }

    // 2. Qwen-style: array of quoted strings with missing commas.
    //    Extract every quoted string inside the brackets in order.
    let strings: Vec<String> = string_re
        .captures_iter(array_text)
        .map(|c| c.get(1).unwrap().as_str().to_string())
        .collect();
    if !strings.is_empty() {
        let mut results = Vec::new();
        for mut s in strings {
            if s.contains('\\') {
                // Properly unescape JSON escapes; never mangle valid
                // non-ASCII text that merely contains a backslash.
                if let Ok(Value::String(unescaped)) =
                    serde_json::from_str::<Value>(&format!("\"{}\"", s))
                {
                    s = unescaped;
                }
            }
            results.push(s.trim().to_string());
        }
        return Some(results);
    }

    None
}

/// Port of _line_fallback. Last-resort: extract all lines from LLM response.
/// Preserves whatever the LLM generated (English, Indonesian, or mixed).
/// Never overwrites already-extracted translations.
fn line_fallback(content: &str, expected_count: usize) -> Vec<String> {
    static NUM_PREFIX_RE: OnceLock<Regex> = OnceLock::new();
    let num_prefix_re =
        NUM_PREFIX_RE.get_or_init(|| Regex::new(r"^\d+[.):]\s*").unwrap());

    let lines: Vec<&str> = content
        .split('\n')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return Vec::new();
    }

    // If lines have numbering, extract the text after the number
    let numbered: Vec<String> = lines
        .iter()
        .filter_map(|&line| {
            let caps = numbered_re().captures(line)?;
            let text = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
            if text.is_empty() {
                None
            } else {
                Some(text.to_string())
            }
        })
        .collect();

    if !numbered.is_empty() {
        // We got numbered lines — use them as-is, even if count differs
        if numbered.len() >= expected_count {
            return numbered[..expected_count].to_vec();
        }
        return numbered;
    }

    // No numbering found — try to extract raw text lines
    // Strip JSON artifacts and stray numbers
    let cleaned: Vec<String> = lines
        .iter()
        .filter_map(|&line| {
            let line = num_prefix_re.replace_all(line, "");
            let line = line.trim_matches(|c| c == '"' || c == ',' || c == '[' || c == ']');
            if line.is_empty() {
                None
            } else {
                Some(line.to_string())
            }
        })
        .collect();

    if cleaned.is_empty() {
        return Vec::new();
    }

    if cleaned.len() >= expected_count {
        return cleaned[..expected_count].to_vec();
    }
    cleaned
}
