use diesel::prelude::*;

use super::schema::{correction_pairs, dictionary, sessions, snippets};

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
}

// ── Correction pairs (homophone calibration data) ────────────────────────────

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
