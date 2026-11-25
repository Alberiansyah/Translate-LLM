//! Decoding + parsing of subtitle files into events with byte spans
//! (template splice design; see mod.rs).
//!
//! Parity target: parser.load_subtitle (pysubs2-based). Where pysubs2's real
//! readers differ from the block-based algorithm prescribed for this rewrite
//! (SRT/VTT block parsing, [Events]-gated ASS parsing, Format:-driven field
//! count), the prescribed algorithm is implemented and the divergence is
//! noted inline.

use super::{Span, SubFormat, SubtitleEvent, SubtitleFile, SubtitleLine};
use crate::subtitle::tags;
use std::path::Path;

/// Port of parser.load_subtitle: decode (utf-8-sig -> utf-8 -> utf-16 ->
/// cp1252 -> latin-1), parse into events, extract tags for ASS/SSA, clean for
/// SRT/VTT, and drop events whose cleaned text is empty.
pub fn load_subtitle(path: &Path) -> Result<SubtitleFile, String> {
    let raw = std::fs::read(path)
        .map_err(|e| format!("cannot read subtitle file {}: {e}", path.display()))?;

    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let format = SubFormat::from_extension(&ext)
        .ok_or_else(|| format!("unsupported subtitle extension: .{ext} (expected srt, ass, ssa or vtt)"))?;

    let (source, encoding) = decode(&raw)?;

    let mut events = match format {
        SubFormat::Srt => parse_srt(&source),
        SubFormat::Ass | SubFormat::Ssa => parse_ass(&source),
        SubFormat::Vtt => parse_vtt(&source),
    };
    let lines = build_lines(&mut events, format);

    Ok(SubtitleFile {
        format,
        encoding,
        source,
        events,
        lines,
    })
}

// ---------------------------------------------------------------------------
// Decoding chain (parser.py lines ~128-145)
// ---------------------------------------------------------------------------

/// Decode `raw` trying, in order and first strict success wins:
/// utf-8-sig, utf-8, utf-16 (BOM-driven endianness, else LE), cp1252,
/// latin-1. Returns the text and the encoding label.
fn decode(raw: &[u8]) -> Result<(String, &'static str), String> {
    // a. utf-8-sig: Python's chain tries codecs "utf-8-sig" first. That
    //    codec strips an optional BOM and is otherwise plain strict UTF-8,
    //    so ANY valid UTF-8 file (BOM or not) decodes in this step and gets
    //    the "utf-8-sig" label — parser.py's later "utf-8" step can never be
    //    reached for valid UTF-8 input. Replicate exactly (a + b collapsed).
    let body = raw.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(raw);
    if let Ok(s) = std::str::from_utf8(body) {
        return Ok((s.to_owned(), "utf-8-sig"));
    }

    // c. utf-16, strict, ONLY with a BOM. Parity: pysubs2 opens the file via
    //    TextIOWrapper, and Python's utf-16 STREAM decoder raises
    //    UnicodeError("Stream does not start with BOM") for BOM-less input —
    //    so a BOM-less cp1252 file must fall through to cp1252, not be
    //    swallowed as UTF-16LE garbage.
    if raw.len() >= 2 && raw[0] == 0xFF && raw[1] == 0xFE {
        let (s, had_errors) = encoding_rs::UTF_16LE.decode_with_bom_removal(raw);
        if !had_errors {
            return Ok((s.into_owned(), "utf-16"));
        }
    } else if raw.len() >= 2 && raw[0] == 0xFE && raw[1] == 0xFF {
        let (s, had_errors) = encoding_rs::UTF_16BE.decode_with_bom_removal(raw);
        if !had_errors {
            return Ok((s.into_owned(), "utf-16"));
        }
    }

    // d. cp1252, strict: encoding_rs reports errors on the 5 undefined bytes
    //    (0x81, 0x8D, 0x8F, 0x90, 0x9D), like Python codecs.cp1252 raising
    //    UnicodeError. No BOM sniffing (Python codecs do not sniff).
    {
        let (s, had_errors) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(raw);
        if !had_errors {
            return Ok((s.into_owned(), "cp1252"));
        }
    }

    // e. latin-1: never fails. encoding_rs follows the WHATWG standard where
    // "latin-1" is an alias of windows-1252, so there is no true latin-1
    // decoder there; Python's latin-1 maps each byte to the code point with
    // the same value, which is exactly what a bytewise char cast does.
    let s: String = raw.iter().map(|&b| b as char).collect();
    Ok((s, "latin-1"))
}

// ---------------------------------------------------------------------------
// Line iteration with byte offsets
// ---------------------------------------------------------------------------

/// Logical lines of `source` as `(byte offset of first char, text without
/// terminator)`. `\r\n`, `\n` and lone `\r` are all line terminators
/// (universal-newline parity with Python text mode).
fn split_lines(source: &str) -> Vec<(usize, &str)> {
    let mut lines = Vec::new();
    let bytes = source.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push((start, &source[start..i]));
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push((start, &source[start..i]));
                i += 1;
                if bytes.get(i) == Some(&b'\n') {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push((start, &source[start..]));
    }
    lines
}

/// Slice `source[start..end]` and normalize `\r\n` / lone `\r` to `\n`
/// (normalization applies to the event text field only; the span keeps the
/// original bytes so untouched regions stay byte-identical).
fn normalize_region(source: &str, start: usize, end: usize) -> String {
    let region = &source[start..end];
    if region.contains('\r') {
        region.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        region.to_owned()
    }
}

/// Blank-line check (parity with block splitting on empty/stripped lines).
fn is_blank(line: &str) -> bool {
    line.trim().is_empty()
}

// ---------------------------------------------------------------------------
// SRT parsing
// ---------------------------------------------------------------------------

/// SRT: blocks separated by blank lines; a block is an optional index line
/// (all digits) + a timing line containing "-->" + zero or more text lines.
/// Blocks without a "-->" line produce no event (SRT has no comments).
fn parse_srt(source: &str) -> Vec<SubtitleEvent> {
    let lines = split_lines(source);
    let mut events = Vec::new();

    let mut i = 0usize;
    while i < lines.len() {
        if is_blank(lines[i].1) {
            i += 1;
            continue;
        }
        // Collect the block [i..j).
        let mut j = i;
        while j < lines.len() && !is_blank(lines[j].1) {
            j += 1;
        }

        if let Some(t) = (i..j).find(|&k| lines[k].1.contains("-->")) {
            if t + 1 < j {
                // Event text = text lines joined with "\n"; the span covers
                // the text lines region up to (not including) the final line
                // terminator, so everything else stays byte-identical.
                let span_start = lines[t + 1].0;
                let (last_off, last_text) = lines[j - 1];
                let span_end = last_off + last_text.len();
                let text = normalize_region(source, span_start, span_end);
                events.push(SubtitleEvent {
                    is_comment: false,
                    text,
                    span: Some(Span { start: span_start, end: span_end }),
                });
            }
            // A timing line with no text lines yields no translatable event.
        }
        // Lines before the timing line (the digit index, if present) are
        // left untouched outside the span.

        i = j;
    }
    events
}

// ---------------------------------------------------------------------------
// VTT parsing
// ---------------------------------------------------------------------------

/// VTT: like SRT but with a WEBVTT header (not a cue) and NOTE comment
/// blocks. Cue timing lines may carry settings after the end timestamp;
/// those stay outside the span.
fn parse_vtt(source: &str) -> Vec<SubtitleEvent> {
    let lines = split_lines(source);
    let mut events = Vec::new();

    let mut i = 0usize;
    while i < lines.len() {
        if is_blank(lines[i].1) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < lines.len() && !is_blank(lines[j].1) {
            j += 1;
        }

        let first = lines[i].1.trim_start();
        let is_note = first == "NOTE"
            || first.starts_with("NOTE ")
            || first.starts_with("NOTE\t");
        if is_note {
            // NOTE block: comment event, never translated, no span.
            let text = lines[i..j]
                .iter()
                .map(|(_, t)| *t)
                .collect::<Vec<_>>()
                .join("\n");
            events.push(SubtitleEvent {
                is_comment: true,
                text,
                span: None,
            });
            i = j;
            continue;
        }

        // The WEBVTT header and STYLE/REGION blocks have no cue timing and
        // produce no event (parity: pysubs2 only creates events for cue
        // lines).
        if let Some(t) = (i..j).find(|&k| lines[k].1.contains("-->")) {
            if t + 1 < j {
                let span_start = lines[t + 1].0;
                let (last_off, last_text) = lines[j - 1];
                let span_end = last_off + last_text.len();
                let text = normalize_region(source, span_start, span_end);
                events.push(SubtitleEvent {
                    is_comment: false,
                    text,
                    span: Some(Span { start: span_start, end: span_end }),
                });
            }
        }

        i = j;
    }
    events
}

// ---------------------------------------------------------------------------
// ASS / SSA parsing
// ---------------------------------------------------------------------------

/// Standard 10-field event order (Layer,Start,End,Style,Name,MarginL,MarginR,
/// MarginV,Effect,Text) assumed when the [Events] section has no Format:
/// line. The Text field is the last one.
const DEFAULT_EVENT_FIELDS: usize = 10;

/// ASS/SSA: track the current `[section]`; inside `[Events]`, read the
/// Format: line for the field count (standard order otherwise) and split
/// Dialogue:/Comment: lines so that the Text field is the remainder after the
/// (n_fields-1)th comma. The event text is kept verbatim (override tags and
/// literal \N intact).
fn parse_ass(source: &str) -> Vec<SubtitleEvent> {
    let lines = split_lines(source);
    let mut events = Vec::new();
    let mut field_count = DEFAULT_EVENT_FIELDS;
    let mut in_events = false;

    for (off, line) in &lines {
        // pysubs2 strips every line before parsing; replicate that for
        // structure detection while keeping original byte offsets for spans.
        let lead = line.len() - line.trim_start().len();
        let core = line.trim();

        if core.starts_with('[') && core.len() >= 2 && core.ends_with(']') {
            let name = &core[1..core.len() - 1];
            in_events = name.eq_ignore_ascii_case("events");
            continue;
        }
        if !in_events {
            continue;
        }

        if let Some(rest) = core.strip_prefix("Format:") {
            let n = rest.split(',').count();
            if n > 0 {
                field_count = n;
            }
            continue;
        }

        let (is_comment, prefix_len) = if core.starts_with("Dialogue:") {
            (false, "Dialogue:".len())
        } else if core.starts_with("Comment:") {
            (true, "Comment:".len())
        } else {
            continue;
        };

        // pysubs2: rest.strip().split(",", n_fields - 1); the Text field is
        // the remainder after the (n_fields-1)th comma and is NOT stripped
        // itself (pysubs2 preserves whitespace in Text). Locate that region
        // in the stripped line, then map back onto the original bytes.
        let rem = &core[prefix_len..];
        let text_start_in_core = match nth_comma(rem, field_count - 1) {
            Some(p) => prefix_len + p + 1,
            None => match rem.rfind(',') {
                // Malformed line with fewer commas than fields-1 (rare):
                // pysubs2's maxsplit then behaves like a plain split, so the
                // text field is everything after the LAST comma.
                Some(p) => prefix_len + p + 1,
                // No comma at all: the whole remainder is the text field.
                None => prefix_len,
            },
        };

        let span_start = off + lead + text_start_in_core;
        let span_end = off + lead + core.len();
        let text = source[span_start..span_end].to_owned();

        events.push(SubtitleEvent {
            is_comment,
            text,
            // Comment events are never translated, so no span is recorded
            // (parity with parser.py skipping is_comment events).
            span: if is_comment {
                None
            } else {
                Some(Span { start: span_start, end: span_end })
            },
        });
    }
    events
}

/// Byte offset within `s` of the n-th comma (n is 1-based); None if `s` has
/// fewer than n commas.
fn nth_comma(s: &str, n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let mut count = 0usize;
    for (i, c) in s.char_indices() {
        if c == ',' {
            count += 1;
            if count == n {
                return Some(i);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Building SubtitleLines
// ---------------------------------------------------------------------------

/// Iterate events in order, skipping comments; extract tags for ASS/SSA,
/// clean for SRT/VTT. Events whose cleaned text is empty are dropped and
/// must never be spliced into, so their span is cleared to None. Comment
/// events keep span None as well.
fn build_lines(events: &mut Vec<SubtitleEvent>, format: SubFormat) -> Vec<SubtitleLine> {
    let mut lines = Vec::new();
    let mut used = vec![false; events.len()];

    for (idx, ev) in events.iter().enumerate() {
        if ev.is_comment {
            continue; // parser.py skips comment events
        }
        let (prefix, cleaned, suffix, auto_close) = if format.has_override_tags() {
            tags::extract_tags(&ev.text)
        } else {
            (String::new(), tags::clean_text(&ev.text), String::new(), String::new())
        };
        if cleaned.is_empty() {
            continue;
        }
        used[idx] = true;
        lines.push(SubtitleLine {
            text: cleaned,
            raw_text: ev.text.clone(),
            prefix_tags: prefix,
            suffix_tags: suffix,
            auto_close_tags: auto_close,
            event: idx,
        });
    }

    for (idx, ev) in events.iter_mut().enumerate() {
        if !used[idx] {
            ev.span = None;
        }
    }

    lines
}
