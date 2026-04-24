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
    }
}
