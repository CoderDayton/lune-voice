// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::mpsc;
use std::time::Duration;

use tauri::{
    AppHandle, Emitter, Listener, Runtime, WebviewWindow, WebviewWindowBuilder,
    ipc::InvokeBody,
    test::{MockRuntime, get_ipc_response, mock_builder, mock_context, noop_assets},
    webview::InvokeRequest,
};

use crate::commands;
use crate::dto::{ActivationModeDto, PillSnapshotDto, PillVisibilityDto};

#[tauri::command]
fn test_main_only(window: WebviewWindow<MockRuntime>) -> Result<&'static str, String> {
    commands::require_window_access(&window, &["main"])?;
    Ok("main")
}

#[tauri::command]
fn test_pill_only(window: WebviewWindow<MockRuntime>) -> Result<&'static str, String> {
    commands::require_window_access(&window, &["pill"])?;
    Ok("pill")
}

#[tauri::command]
fn test_load_pill_snapshot(window: WebviewWindow<MockRuntime>) -> Result<PillSnapshotDto, String> {
    commands::require_window_access(&window, &["pill"])?;
    Ok(PillSnapshotDto {
        service_status: "ready".into(),
        activation_mode: ActivationModeDto::Toggle,
        pill_visibility: PillVisibilityDto::Always,
        pill_locked: false,
    })
}

#[tauri::command]
fn test_emit_pill_settings(
    window: WebviewWindow<MockRuntime>,
    app: AppHandle<MockRuntime>,
) -> Result<(), String> {
    commands::require_window_access(&window, &["pill"])?;
    app.emit_to(
        "pill",
        "lune://pill-settings",
        serde_json::json!({
            "activationMode": "Hold",
            "pillVisibility": "WhenDictating",
        }),
    )
    .map_err(|e| e.to_string())
}

fn build_test_app() -> tauri::App<MockRuntime> {
    mock_builder()
        .invoke_handler(tauri::generate_handler![
            test_main_only,
            test_pill_only,
            test_load_pill_snapshot,
            test_emit_pill_settings
        ])
        .build(mock_context(noop_assets()))
        .expect("failed to build test app")
}

fn build_test_windows<R: Runtime>(app: &tauri::App<R>) -> (WebviewWindow<R>, WebviewWindow<R>) {
    let main = WebviewWindowBuilder::new(app, "main", Default::default())
        .build()
        .expect("failed to build main window");
    let pill = WebviewWindowBuilder::new(app, "pill", Default::default())
        .build()
        .expect("failed to build pill window");
    (main, pill)
}

fn invoke_no_args(webview: &WebviewWindow<MockRuntime>, cmd: &str) -> Result<String, String> {
    invoke_command::<String>(webview, cmd)
}

/// Origin the IPC authority matches invocations against. Windows and Android
/// serve the app over http; everywhere else it is the custom protocol. Getting
/// this wrong makes every command fail the ACL check as "Plugin not found".
const IPC_ORIGIN: &str = if cfg!(any(windows, target_os = "android")) {
    "http://tauri.localhost"
} else {
    "tauri://localhost"
};

fn invoke_command<T: serde::de::DeserializeOwned>(
    webview: &WebviewWindow<MockRuntime>,
    cmd: &str,
) -> Result<T, String> {
    let request = InvokeRequest {
        cmd: cmd.into(),
        callback: tauri::ipc::CallbackFn(0),
        error: tauri::ipc::CallbackFn(1),
        url: IPC_ORIGIN.parse().unwrap(),
        body: InvokeBody::default(),
        headers: Default::default(),
        invoke_key: tauri::test::INVOKE_KEY.to_string(),
    };

    match get_ipc_response(webview, request) {
        Ok(body) => Ok(body.deserialize::<T>().unwrap()),
        Err(error) => Err(serde_json::from_value::<String>(error).unwrap()),
    }
}

#[test]
fn invokes_window_scoped_commands_through_tauri_ipc() {
    let app = build_test_app();
    let (main, pill) = build_test_windows(&app);

    assert_eq!(invoke_no_args(&main, "test_main_only"), Ok("main".into()));
    assert_eq!(
        invoke_no_args(&pill, "test_main_only"),
        Err("window \"pill\" cannot invoke this command".into())
    );

    assert_eq!(invoke_no_args(&pill, "test_pill_only"), Ok("pill".into()));
    assert_eq!(
        invoke_no_args(&main, "test_pill_only"),
        Err("window \"main\" cannot invoke this command".into())
    );
}

#[test]
fn invokes_structured_pill_snapshot_command_through_tauri_ipc() {
    let app = build_test_app();
    let (main, pill) = build_test_windows(&app);

    let snapshot = invoke_command::<PillSnapshotDto>(&pill, "test_load_pill_snapshot")
        .expect("pill snapshot invoke should succeed");
    assert_eq!(snapshot.service_status, "ready");
    assert_eq!(snapshot.activation_mode, ActivationModeDto::Toggle);
    assert_eq!(snapshot.pill_visibility, PillVisibilityDto::Always);

    assert_eq!(
        invoke_command::<PillSnapshotDto>(&main, "test_load_pill_snapshot"),
        Err("window \"main\" cannot invoke this command".into())
    );
}

#[test]
fn emits_targeted_events_only_to_the_pill_window() {
    let app = build_test_app();
    let (main, pill) = build_test_windows(&app);
    let (tx, rx) = mpsc::channel();

    let main_label = main.label().to_string();
    main.listen("lune://pill-state", {
        let tx = tx.clone();
        move |event| {
            tx.send((
                main_label.clone(),
                serde_json::from_str::<String>(event.payload()).unwrap(),
            ))
            .unwrap();
        }
    });

    let pill_label = pill.label().to_string();
    pill.listen("lune://pill-state", move |event| {
        tx.send((
            pill_label.clone(),
            serde_json::from_str::<String>(event.payload()).unwrap(),
        ))
        .unwrap();
    });

    app.emit_to("pill", "lune://pill-state", "processing")
        .expect("failed to emit pill-state");

    let mut received = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_millis(50)) {
        received.push(event);
    }

    assert_eq!(received, vec![("pill".into(), "processing".into())]);
}

#[test]
fn pill_only_command_emits_pill_settings_event_via_ipc() {
    let app = build_test_app();
    let (main, pill) = build_test_windows(&app);
    let (tx, rx) = mpsc::channel();

    main.listen("lune://pill-settings", {
        let tx = tx.clone();
        move |event| {
            tx.send(("main".to_string(), event.payload().to_string()))
                .unwrap();
        }
    });

    pill.listen("lune://pill-settings", move |event| {
        tx.send(("pill".to_string(), event.payload().to_string()))
            .unwrap();
    });

    invoke_command::<()>(&pill, "test_emit_pill_settings")
        .expect("pill emit invoke should succeed");

    let mut received = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_millis(50)) {
        received.push(event);
    }

    assert_eq!(
        received,
        vec![(
            "pill".into(),
            "{\"activationMode\":\"Hold\",\"pillVisibility\":\"WhenDictating\"}".into(),
        )]
    );

    assert_eq!(
        invoke_command::<()>(&main, "test_emit_pill_settings"),
        Err("window \"main\" cannot invoke this command".into())
    );
}
