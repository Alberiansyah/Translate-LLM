//! Subtitle parsing/saving with byte-faithful round-trip ("template splice").
//!
//! Parity target: app/translator/parser.py (pysubs2-based).
//! Divergence (intentional, documented in the rewrite report): instead of
//! re-serializing a parsed subtitle model, we keep the original decoded file
//! text (`source`) and record, per translatable event, the byte span of its
//! text region. Saving splices translated text into those spans, leaving
//! everything else byte-identical. Output format always equals the input
//! extension (same as parser.SAVE_FORMAT_MAP).

pub mod formats;
pub mod save;
pub mod tags;

use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubFormat {
    Srt,
    Ass,
    Ssa,
    Vtt,
}

impl SubFormat {
    pub fn from_extension(ext: &str) -> Option<SubFormat> {
        match ext.to_ascii_lowercase().as_str() {
            "srt" => Some(SubFormat::Srt),
            "ass" => Some(SubFormat::Ass),
            "ssa" => Some(SubFormat::Ssa),
            "vtt" => Some(SubFormat::Vtt),
            _ => None,
        }
    }

    /// True for formats whose text carries ASS override tags ({\...}).
    pub fn has_override_tags(self) -> bool {
        matches!(self, SubFormat::Ass | SubFormat::Ssa)
    }
}

/// Byte span of a translatable text region inside `SubtitleFile::source`.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// One parsed event (cue/dialogue line/comment).
#[derive(Clone, Debug)]
pub struct SubtitleEvent {
    /// ASS "Comment:" lines and VTT NOTE blocks; SRT has none.
    pub is_comment: bool,
    /// Raw text of the event as pysubs2's event.text would expose it:
    /// override tags kept, line breaks as "\n" (SRT/VTT) or "\N" (ASS/SSA).
    pub text: String,
    /// Byte span of the text region in `source` (None for comments and for
    /// events whose cleaned text is empty — those are never translated).
    pub span: Option<Span>,
}

/// One translatable line (cleaned text + ASS tag scaffolding).
#[derive(Clone, Debug)]
pub struct SubtitleLine {
    /// Cleaned text sent to the LLM (tags stripped, breaks normalized).
    pub text: String,
    pub raw_text: String,
    pub prefix_tags: String,
    pub suffix_tags: String,
    pub auto_close_tags: String,
    /// Index into `SubtitleFile::events`.
    pub event: usize,
}

#[derive(Debug)]
pub struct SubtitleFile {
    pub format: SubFormat,
    /// Encoding label used to decode the file (parity with parser.py chain).
    pub encoding: &'static str,
    /// Original file text, decoded and BOM-stripped.
    pub source: String,
    pub events: Vec<SubtitleEvent>,
    pub lines: Vec<SubtitleLine>,
}

impl SubtitleFile {
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
}

/// Port of parser.load_subtitle: decode (utf-8-sig → utf-8 → utf-16 →
/// cp1252 → latin-1), parse into events, extract tags for ASS/SSA, clean for
/// SRT/VTT, and drop events whose cleaned text is empty.
pub fn load_subtitle(path: &Path) -> Result<SubtitleFile, String> {
    formats::load_subtitle(path)
}

/// Port of parser.save_subtitle: splice translated lines into the recorded
/// spans and write atomically (tmp file + replace). For ASS/SSA the spliced
/// text is prefix_tags + new_text + auto_close_tags + suffix_tags.
pub fn save_subtitle(
    sub: &SubtitleFile,
    translated_lines: &[String],
    output_path: &Path,
) -> Result<std::path::PathBuf, String> {
    save::save_subtitle(sub, translated_lines, output_path)
}
