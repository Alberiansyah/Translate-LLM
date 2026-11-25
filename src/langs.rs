//! Language list, ported verbatim from app/api/routes.py LANGUAGES.
//! Names are sent to the LLM prompt verbatim, so the dropdown value is the name.

pub const LANGUAGES: &[(&str, &str)] = &[
    // Most common first
    ("en", "English"), ("id", "Indonesian"), ("ms", "Malay"),
    ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
    ("th", "Thai"), ("vi", "Vietnamese"), ("tl", "Filipino"),
    ("ar", "Arabic"), ("hi", "Hindi"), ("bn", "Bengali"),
    ("pt", "Portuguese"), ("es", "Spanish"), ("fr", "French"),
    ("de", "German"), ("it", "Italian"), ("ru", "Russian"),
    ("tr", "Turkish"), ("pl", "Polish"), ("nl", "Dutch"),
    ("sv", "Swedish"), ("no", "Norwegian"), ("da", "Danish"),
    ("fi", "Finnish"), ("cs", "Czech"), ("sk", "Slovak"),
    ("hu", "Hungarian"), ("ro", "Romanian"), ("bg", "Bulgarian"),
    ("hr", "Croatian"), ("sr", "Serbian"), ("uk", "Ukrainian"),
    ("el", "Greek"), ("he", "Hebrew"), ("fa", "Persian"),
    ("sw", "Swahili"), ("ta", "Tamil"), ("te", "Telugu"),
    ("ml", "Malayalam"), ("my", "Myanmar"), ("km", "Khmer"),
    ("lo", "Lao"), ("ka", "Georgian"), ("am", "Amharic"),
    ("ne", "Nepali"), ("si", "Sinhala"), ("ur", "Urdu"),
    // Extended list (alphabetical by code)
    ("af", "Afrikaans"), ("az", "Azerbaijani"), ("be", "Belarusian"),
    ("bs", "Bosnian"), ("ca", "Catalan"), ("ceb", "Cebuano"),
    ("co", "Corsican"), ("cy", "Welsh"), ("eo", "Esperanto"),
    ("et", "Estonian"), ("eu", "Basque"), ("fy", "Frisian"),
    ("ga", "Irish"), ("gd", "Scottish Gaelic"), ("gl", "Galician"),
    ("gu", "Gujarati"), ("ha", "Hausa"), ("haw", "Hawaiian"),
    ("hmn", "Hmong"), ("ht", "Haitian Creole"), ("hy", "Armenian"),
    ("ig", "Igbo"), ("is", "Icelandic"), ("iu", "Inuktitut"),
    ("jv", "Javanese"), ("kk", "Kazakh"), ("kn", "Kannada"),
    ("ku", "Kurdish"), ("ky", "Kyrgyz"), ("la", "Latin"),
    ("lb", "Luxembourgish"), ("lt", "Lithuanian"), ("lv", "Latvian"),
    ("mg", "Malagasy"), ("mi", "Maori"), ("mk", "Macedonian"),
    ("mn", "Mongolian"), ("mr", "Marathi"), ("mt", "Maltese"),
    ("ny", "Chichewa"), ("or", "Odia"), ("pa", "Punjabi"),
    ("ps", "Pashto"), ("sd", "Sindhi"), ("sm", "Samoan"),
    ("sn", "Shona"), ("so", "Somali"), ("sq", "Albanian"),
    ("st", "Sesotho"), ("su", "Sundanese"), ("tg", "Tajik"),
    ("ts", "Tsonga"), ("tt", "Tatar"), ("ug", "Uyghur"),
    ("uz", "Uzbek"), ("xh", "Xhosa"), ("yi", "Yiddish"),
    ("yo", "Yoruba"), ("zu", "Zulu"),
];

/// The dropdown sends the language NAME; validate against the names.
pub fn is_valid_lang_name(name: &str) -> bool {
    LANGUAGES.iter().any(|(_, n)| *n == name)
}
