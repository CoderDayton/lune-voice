use diesel::prelude::*;

use super::schema::{correction_pairs, custom_styles, dictionary, sessions, snippets};

// ── Custom styles ────────────────────────────────────────────────────────────

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = custom_styles)]
#[allow(dead_code)]
pub struct CustomStyleRow {
    pub id: String,
    pub name: String,
    pub description: String,
    pub style_instructions: String,
    /// JSON-serialized `FormatToggles`.
    pub formatting: String,
    /// JSON-serialized `Vec<(String, String)>` of few-shot pairs.
    pub few_shot_pairs: String,
    pub few_shot_dirty: i32,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Insertable)]
#[diesel(table_name = custom_styles)]
pub struct NewCustomStyle<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub description: &'a str,
    pub style_instructions: &'a str,
    pub formatting: &'a str,
    pub few_shot_pairs: &'a str,
    pub few_shot_dirty: i32,
}

// ── Dictionary ───────────────────────────────────────────────────────────────

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = dictionary)]
#[allow(dead_code)]
pub struct DictionaryRow {
    pub id: i32,
    pub phrase: String,
    pub replacement: String,
    pub priority: i32,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Insertable)]
#[diesel(table_name = dictionary)]
pub struct NewDictionaryEntry<'a> {
    pub phrase: &'a str,
    pub replacement: &'a str,
    pub priority: i32,
}

// ── Snippets ─────────────────────────────────────────────────────────────────

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = snippets)]
#[allow(dead_code)]
pub struct SnippetRow {
    pub id: String,
    pub trigger_phrase: String,
    pub body: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Insertable)]
#[diesel(table_name = snippets)]
pub struct NewSnippet<'a> {
    pub id: &'a str,
    pub trigger_phrase: &'a str,
    pub body: &'a str,
}

// ── Sessions ─────────────────────────────────────────────────────────────────

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = sessions)]
#[allow(dead_code)]
pub struct SessionRow {
    pub id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub style: String,
    pub backend: String,
    pub status: String,
    pub transcript: Option<String>,
    pub created_at: String,
}

#[derive(Insertable)]
#[diesel(table_name = sessions)]
pub struct NewSession<'a> {
    pub id: &'a str,
    pub started_at: &'a str,
    pub ended_at: Option<&'a str>,
    pub style: &'a str,
    pub backend: &'a str,
    pub status: &'a str,
    pub transcript: Option<&'a str>,
    /// Maintained alongside `transcript` so `session_stats` never has to read
    /// the text back to count words.
    pub word_count: i32,
}

// ── Correction pairs (raw → final history) ────────────────────────────

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = correction_pairs)]
#[allow(dead_code)]
pub struct CorrectionPairRow {
    pub id: i32,
    pub session_id: String,
    pub raw: String,
    pub final_text: String,
    pub created_at: String,
}

#[derive(Insertable)]
#[diesel(table_name = correction_pairs)]
pub struct NewCorrectionPair<'a> {
    pub session_id: &'a str,
    pub raw: &'a str,
    pub final_text: &'a str,
    pub created_at: &'a str,
}
