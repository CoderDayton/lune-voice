//! Tests for how a found update is announced.
//!
//! Delivery itself is not assertable — every desktop backend hands the
//! notification to a spawned task and returns `Ok` regardless (see
//! `notify_update_available`) — so these cover the two things that *are*: the
//! text the user reads, and the policy deciding which announcement a given
//! check gets.

use tauri::test::{MockRuntime, mock_builder, mock_context, noop_assets};

use crate::update::{CheckMode, notify_update_available, update_notification_text};

const NEW: &str = "1.4.0";
const CURRENT: &str = "1.3.2";

fn app() -> tauri::App<MockRuntime> {
    mock_builder()
        .plugin(tauri_plugin_notification::init())
        .build(mock_context(noop_assets()))
        .expect("failed to build notification test app")
}

#[test]
fn the_notification_offers_the_new_version_and_names_the_installed_one() {
    let (title, body) = update_notification_text(NEW, CURRENT);

    // Both are versions and either order reads plausibly, so pin which is
    // which: a swap would announce the version the user already has.
    assert!(title.contains(NEW), "title should offer {NEW}: {title}");
    assert!(
        !title.contains(CURRENT),
        "the title must not offer the version already installed: {title}"
    );
    assert!(
        body.contains(CURRENT),
        "body should name the installed version {CURRENT}: {body}"
    );
}

#[test]
#[ignore = "posts a real notification through the desktop plugin; run manually with --include-ignored"]
fn posting_an_update_notification_is_accepted_by_the_plugin() {
    // Exercises the real builder chain against a real plugin instance: catches
    // a renamed builder method or a desktop permission requirement appearing
    // in a future plugin version. It does *not* prove anything was displayed.
    let app = app();
    assert!(
        notify_update_available(app.handle(), NEW, CURRENT),
        "the notification plugin should accept a well-formed update notice"
    );
}

#[test]
fn a_requested_check_answers_in_the_app_and_reports_non_events() {
    // The user is waiting on an answer, so every outcome gets one — a control
    // that silently does nothing is indistinguishable from a broken one.
    assert!(CheckMode::Requested.reports_non_events());
    assert!(!CheckMode::Requested.announces_by_notification());
}

#[test]
fn a_launch_check_notifies_and_stays_silent_about_non_events() {
    // Nobody asked. Seizing focus 20 seconds into a session, or reporting
    // "offline at login" as an error, would both be worse than saying nothing.
    assert!(CheckMode::OnLaunch.announces_by_notification());
    assert!(!CheckMode::OnLaunch.reports_non_events());
}
