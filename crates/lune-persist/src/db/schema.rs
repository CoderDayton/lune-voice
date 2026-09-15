// SPDX-License-Identifier: GPL-3.0-or-later

diesel::table! {
    dictionary (id) {
        id -> Integer,
        phrase -> Text,
        replacement -> Text,
        priority -> Integer,
        created_at -> Text,
        updated_at -> Text,
    }
}

diesel::table! {
    custom_styles (id) {
        id -> Text,
        name -> Text,
        description -> Text,
        style_instructions -> Text,
        formatting -> Text,
        few_shot_pairs -> Text,
        few_shot_dirty -> Integer,
        created_at -> Text,
        updated_at -> Text,
    }
}

diesel::table! {
    snippets (id) {
        id -> Text,
        trigger_phrase -> Text,
        body -> Text,
        created_at -> Text,
        updated_at -> Text,
    }
}

diesel::table! {
    sessions (id) {
        id -> Text,
        started_at -> Text,
        ended_at -> Nullable<Text>,
        style -> Text,
        backend -> Text,
        status -> Text,
        transcript -> Nullable<Text>,
        created_at -> Text,
        /// Words in `transcript`, maintained on insert/update. NULL only
        /// before the one-time backfill for pre-migration rows.
        word_count -> Nullable<Integer>,
    }
}

diesel::table! {
    correction_pairs (id) {
        id -> Integer,
        session_id -> Text,
        raw -> Text,
        final_text -> Text,
        created_at -> Text,
    }
}
