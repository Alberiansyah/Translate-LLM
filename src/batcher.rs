//! Port of app/translator/context_batcher.py.
//!
//! Python shares references to SubtitleLine; Rust clones them instead
//! (cheapest correct equivalent here).

use crate::subtitle::SubtitleLine;

/// One context batch: a slice of lines plus the texts sent to the LLM.
pub struct Batch {
    pub batch_index: usize,
    pub lines: Vec<SubtitleLine>,
    pub original_texts: Vec<String>,
}

impl Batch {
    pub fn size(&self) -> usize {
        self.lines.len()
    }
}

/// Groups subtitle lines into context batches of `batch_size` lines.
pub struct ContextBatcher {
    pub batch_size: usize,
}

impl ContextBatcher {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Non-overlapping batches in line order; the last batch may be shorter.
    pub fn create_batches(&self, lines: &[SubtitleLine]) -> Vec<Batch> {
        if lines.is_empty() {
            return Vec::new();
        }

        // Defend against a zero batch_size (Python would raise on range step 0).
        let size = self.batch_size.max(1);

        let mut batches = Vec::new();
        for (batch_idx, chunk) in lines.chunks(size).enumerate() {
            let batch_lines: Vec<SubtitleLine> = chunk.to_vec();
            let original_texts: Vec<String> = batch_lines.iter().map(|l| l.text.clone()).collect();

            batches.push(Batch {
                batch_index: batch_idx,
                lines: batch_lines,
                original_texts,
            });
        }

        batches
    }
}
