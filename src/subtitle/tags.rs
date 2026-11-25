//! Port of parser.py ASS/SSA tag handling: `TAG_PATTERN` / `extract_tags` /
//! `clean_text`.
//!
//! Python indexes are code points; here we work with byte offsets from the
//! `regex` crate. Every match and every slice of a `&str` is on a char
//! boundary, so no multi-byte slicing panics can occur.

use regex::Regex;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// parser.TAG_PATTERN: an ASS override tag such as `{\i1}` or `{\pos(1,2)}`.
fn tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\{\\[^}]*\}").unwrap())
}

/// parser.OPEN_PATTERN: `{\<type>1}` for i/b/u/s.
fn open_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\{\\([ibus])1\}").unwrap())
}

/// parser.CLOSE_PATTERN: `{\<type>0}` for i/b/u/s.
fn close_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\{\\([ibus])0\}").unwrap())
}

/// HTML-ish tag pattern used by parser.clean_text (SRT/VTT).
fn html_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[^>]+>").unwrap())
}

/// Python `re.search(r"[^\s\\]", s)`: true when `s` contains at least one
/// character that is neither whitespace nor a backslash, i.e. visible text.
fn has_visible(s: &str) -> bool {
    s.chars().any(|c| !(c.is_whitespace() || c == '\\'))
}

/// parser.OPEN_CLOSE_PAIRS close side: the explicit close tag per tag type.
fn close_tag(tag_type: &str) -> Option<&'static str> {
    match tag_type {
        "i" => Some(r"{\i0}"),
        "b" => Some(r"{\b0}"),
        "u" => Some(r"{\u0}"),
        "s" => Some(r"{\s0}"),
        _ => None,
    }
}

/// Port of parser.extract_tags.
///
/// Returns `(prefix_tags, clean_text, suffix_tags, auto_close_tags)`.
pub fn extract_tags(raw: &str) -> (String, String, String, String) {
    let re = tag_re();
    let all_tags: Vec<&str> = re.find_iter(raw).map(|m| m.as_str()).collect();

    if all_tags.is_empty() {
        return (String::new(), raw.trim().to_owned(), String::new(), String::new());
    }

    let visible_chars: Cow<str> = re.replace_all(raw, "");
    let has_text = visible_chars.trim() != "";

    // Prefix tags = tags that appear before any visible (non-whitespace,
    // non-backslash) text.
    let mut prefix_tags = String::new();
    if has_text {
        for tag in &all_tags {
            // Python raw.index(tag): first occurrence of the tag text.
            let tag_pos = raw.find(tag).unwrap_or(0);
            let before = re.replace_all(&raw[..tag_pos], "");
            if has_visible(&before) {
                break;
            }
            prefix_tags.push_str(tag);
        }
    }

    // Suffix tags = tags that appear after the last visible text character.
    let mut suffix_tags = String::new();
    if has_text {
        // Scan for the last non-whitespace character OUTSIDE any tag so the
        // digits/letters inside tags (e.g. the "0" in {\i0}) are not counted.
        let last_visible = last_visible_index(raw);
        for tag in all_tags.iter().rev() {
            let tag_pos = raw.find(tag).unwrap_or(0);
            // Between-region: from just past the last visible character up to
            // the tag (Python raw[last_visible_idx + 1:tag_pos]; here the +1
            // advances past the whole character, staying on a char boundary).
            let between: Cow<str> = match last_visible {
                Some(idx) => {
                    if tag_pos <= idx {
                        break;
                    }
                    let after = idx + raw[idx..].chars().next().unwrap_or(' ').len_utf8();
                    re.replace_all(&raw[after..tag_pos], "")
                }
                // No visible char anywhere (Python last_visible_idx == -1):
                // the between-region starts at the beginning of the string.
                None => re.replace_all(&raw[..tag_pos], ""),
            };
            if has_visible(&between) {
                break;
            }
            suffix_tags = format!("{tag}{suffix_tags}");
        }
    }

    let clean = re.replace_all(raw, "").trim().to_owned();

    // Auto-close only tags that were opened but never explicitly closed.
    // BTreeSet iterates in sorted order, matching Python sorted(opens - closes).
    let mut opens: BTreeSet<&str> = BTreeSet::new();
    let mut closes: BTreeSet<&str> = BTreeSet::new();
    for c in open_re().captures_iter(raw) {
        opens.insert(c.get(1).unwrap().as_str());
    }
    for c in close_re().captures_iter(raw) {
        closes.insert(c.get(1).unwrap().as_str());
    }

    let mut auto_close = String::new();
    for tag_type in opens.difference(&closes) {
        if let Some(close) = close_tag(tag_type) {
            auto_close.push_str(close);
        }
    }

    (prefix_tags, clean, suffix_tags, auto_close)
}

/// Byte index of the last non-whitespace character that lies OUTSIDE any tag
/// (port of parser.py's `TAG_PATTERN.match(raw, pos)` walk). Returns None
/// when there is no visible character at all.
///
/// `find_iter` yields leftmost, non-overlapping matches, which is equivalent
/// to Python's try-match-at-each-position walk.
fn last_visible_index(raw: &str) -> Option<usize> {
    let mut last: Option<usize> = None;
    let mut cursor = 0usize;
    for m in tag_re().find_iter(raw) {
        for (i, c) in raw[cursor..m.start()].char_indices() {
            if !c.is_whitespace() {
                last = Some(cursor + i);
            }
        }
        cursor = m.end();
    }
    for (i, c) in raw[cursor..].char_indices() {
        if !c.is_whitespace() {
            last = Some(cursor + i);
        }
    }
    last
}

/// Port of parser.clean_text (used for SRT/VTT): remove ASS override tags and
/// HTML-ish tags, turn line-break markers (`\N`, real newlines) into spaces,
/// and trim.
pub fn clean_text(text: &str) -> String {
    let cleaned = tag_re().replace_all(text, "");
    let cleaned = html_re().replace_all(&cleaned, "");
    // Normalize pysubs2 line-break markers (\N) to spaces so the LLM sees a
    // single continuous line of text.
    let cleaned = cleaned.replace("\\N", " ").replace('\n', " ");
    cleaned.trim().to_owned()
}
