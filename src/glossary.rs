//! Port of app/translator/glossary.py.
//!
//! Python has a `Glossary` wrapper (add/remove/to_dict/from_dict); here the
//! consumers work directly on `Vec<GlossaryEntry>` — serialization via serde
//! covers to_dict/from_dict, and defensive skipping of malformed items happens
//! where the JSON is decoded into the request models.

/// Port of glossary.GlossaryEntry.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GlossaryEntry {
    pub source: String,
    pub target: String,
    #[serde(default = "default_case_sensitive")]
    pub case_sensitive: bool,
}

fn default_case_sensitive() -> bool {
    true
}
