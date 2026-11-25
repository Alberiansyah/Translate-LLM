//! Saving: splice translated text into the recorded byte spans of the
//! original file text (template splice; see mod.rs) and write atomically.
//!
//! Parity target: parser.save_subtitle. Everything outside the recorded text
//! regions (timing lines, styles, script info, NOTE blocks, ...) stays
//! byte-identical, and the output format always equals the input extension.

use super::{Span, SubtitleFile};
use std::path::{Path, PathBuf};

/// Port of parser.save_subtitle: for each line index `i` below
/// `translated.len()`, splice into `events[line.event]`:
/// - ASS/SSA: prefix_tags + translated[i] + auto_close_tags + suffix_tags
/// - SRT/VTT: translated[i] alone
///
/// Spans are verified to be non-overlapping and in file order before
/// splicing. The result is written UTF-8 atomically (tmp file + rename).
pub fn save_subtitle(
    sub: &SubtitleFile,
    translated: &[String],
    out: &Path,
) -> Result<PathBuf, String> {
    // Extra translations beyond the parsed lines are ignored, matching
    // parser.py (its event loop simply runs out of events).
    let count = translated.len().min(sub.lines.len());

    let mut replacements: Vec<(Span, String)> = Vec::with_capacity(count);
    for i in 0..count {
        let line = &sub.lines[i];
        let ev_idx = line.event;
        if ev_idx >= sub.events.len() {
            return Err(format!("line {i} references unknown event {ev_idx}"));
        }
        let span = sub.events[ev_idx]
            .span
            .ok_or_else(|| format!("event {ev_idx} has no text span to splice into"))?;

        let replacement = if sub.format.has_override_tags() {
            // The ASS text field cannot contain real newlines; pysubs2's
            // writer would escape them, so convert any "\n" to the literal
            // "\N" line-break marker before splicing.
            let new_text = translated[i].replace('\n', "\\N");
            format!(
                "{}{}{}{}",
                line.prefix_tags, new_text, line.auto_close_tags, line.suffix_tags
            )
        } else {
            // If the cue's text region used \r\n line breaks, keep that
            // style for multi-line translations so the output stays
            // internally consistent.
            match_region_newlines(&sub.source, span, &translated[i])
        };
        replacements.push((span, replacement));
    }

    // Spans are produced in file order by load_subtitle; verify rather than
    // assume (a violation would silently corrupt the output).
    for pair in replacements.windows(2) {
        if pair[0].0.end > pair[1].0.start {
            return Err(
                "internal error: text spans overlap or are out of file order".to_string()
            );
        }
    }

    // Splice all replacements into the source in one pass.
    let mut result = String::with_capacity(sub.source.len());
    let mut cursor = 0usize;
    for (span, replacement) in &replacements {
        result.push_str(&sub.source[cursor..span.start]);
        result.push_str(replacement);
        cursor = span.end;
    }
    result.push_str(&sub.source[cursor..]);

    // Atomic write (parity with parser.py: mkdir parent, write "<out>.tmp",
    // rename over the output; remove the tmp file on any failure). Output is
    // UTF-8 like Python's encoding="utf-8".
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!("cannot create directory {}: {e}", parent.display())
            })?;
        }
    }
    let file_name = out
        .file_name()
        .ok_or_else(|| format!("invalid output path: {}", out.display()))?;
    let tmp = out.with_file_name(format!("{}.tmp", file_name.to_string_lossy()));

    if let Err(e) = std::fs::write(&tmp, result.as_bytes()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("cannot write temporary file {}: {e}", tmp.display()));
    }
    // On Windows std::fs::rename uses MoveFileEx with
    // MOVEFILE_REPLACE_EXISTING, so it overwrites an existing output file
    // like Python's os.replace.
    if let Err(e) = std::fs::rename(&tmp, out) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "cannot move {} over {}: {e}",
            tmp.display(),
            out.display()
        ));
    }

    Ok(out.to_path_buf())
}

/// Convert "\n" in `replacement` to "\r\n" when the region being replaced
/// used CRLF line breaks, so a multi-line translation keeps the file's
/// line-ending style inside the cue text block.
fn match_region_newlines(source: &str, span: Span, replacement: &str) -> String {
    if replacement.contains('\n') && source[span.start..span.end].contains("\r\n") {
        replacement.replace("\n", "\r\n")
    } else {
        replacement.to_owned()
    }
}
