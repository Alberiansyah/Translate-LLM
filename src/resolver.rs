//! Port of app/translator/context_resolver.py (minimal context-aware
//! pronoun resolution).
//!
//! The code does NOT hardcode gender maps. It just extracts character names
//! and injects a brief context hint so the LLM can resolve "Dia" → He/She/They.

use regex::Regex;
use std::sync::OnceLock;

/// Words that start with capital but aren't character names
const NON_NAMES: &[&str] = &[
    "diam", "dia", "itu", "apa", "siapa", "kenapa", "dimana", "mana", "bagaimana", "seperti",
    "karena", "oleh", "untuk", "dari", "dengan", "pada", "juga", "sudah", "baru", "yang", "ini",
    "atau", "tetapi", "tak", "tidak", "saya", "kau", "masih",
];

const COMMON_WORDS: &[&str] = &[
    "the", "this", "that", "when", "where", "what", "who", "because", "before", "after", "but",
    "and", "or", "if", "then", "will", "would", "could", "should", "is", "are", "was", "were",
    "has", "have", "had", "not", "no", "yes", "can", "may", "might", "just", "been", "only",
    "also", "very", "much", "more", "most", "well", "here", "there", "still", "even",
];

fn name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Z][a-zA-Z]+").unwrap())
}

/// Port of ContextResolver._extract_names: extract potential character names
/// from a subtitle line.
fn extract_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for w in name_re().find_iter(text) {
        let w = w.as_str();
        let w_lower = w.to_ascii_lowercase();
        if w.chars().count() < 2
            || NON_NAMES.contains(&w_lower.as_str())
            || COMMON_WORDS.contains(&w_lower.as_str())
        {
            continue;
        }
        names.push(w.to_string());
    }
    names
}

/// Port of ContextResolver.build_prompt_hint: build a brief dialogue context
/// hint for the LLM prompt.
///
/// Extracts character names and 'Dia' references.
/// Returns empty string if nothing useful found.
pub fn build_prompt_hint(texts: &[String]) -> String {
    let mut characters: Vec<String> = Vec::new();
    let mut dia_lines: Vec<usize> = Vec::new();

    for (idx, text) in texts.iter().enumerate() {
        for name in extract_names(text) {
            if !characters.contains(&name) {
                characters.push(name);
            }
        }
        if text.to_lowercase().contains("dia") {
            dia_lines.push(idx);
        }
    }

    if characters.is_empty() && dia_lines.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    if !characters.is_empty() {
        // Python: sorted(characters)[:8] — byte/codepoint sort matches here.
        characters.sort();
        let char_list: Vec<String> = characters.into_iter().take(8).collect();
        parts.push(format!("Characters: {}", char_list.join(", ")));
    }
    if !dia_lines.is_empty() {
        let lines: Vec<String> = dia_lines
            .iter()
            .take(5)
            .map(|i| (i + 1).to_string())
            .collect();
        parts.push(format!("'Dia' appears in lines: {}", lines.join(", ")));
        parts.push(
            "Note: 'Dia' = He/She/They depending on context. Read surrounding dialogue to determine."
                .to_string(),
        );
    }

    // Two-space indented parts, prefixed by the DIALOGUE CONTEXT header
    // (which already carries its own leading blank lines).
    let body: Vec<String> = parts.iter().map(|p| format!("  {}", p)).collect();
    format!("\n\nDIALOGUE CONTEXT:\n{}", body.join("\n"))
}
