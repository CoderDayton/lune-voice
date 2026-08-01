//! Updater tests against a local HTTP endpoint.
//!
//! These cover the part that carries the trust: a manifest is honoured only if
//! the payload it points at verifies against the pinned public key. The install
//! step is deliberately not exercised — it would replace the running app — but
//! everything up to and including signature verification is real, using a
//! fixture signed by a throwaway key (see `tests/fixtures/`, regenerate with
//! `bunx tauri signer generate` + `sign` if the payload ever changes).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use tauri::test::{MockRuntime, mock_builder, mock_context, noop_assets};
use tauri_plugin_updater::UpdaterExt;

const PAYLOAD: &[u8] = include_bytes!("../tests/fixtures/update-payload.bin");
const SIGNATURE: &str = include_str!("../tests/fixtures/update-payload.bin.sig");
const PUBKEY: &str = include_str!("../tests/fixtures/test-pubkey.txt");
/// A valid key that signed nothing here — proves verification is key-specific
/// rather than merely checking the signature is well-formed.
const OTHER_PUBKEY: &str = include_str!("../tests/fixtures/other-pubkey.txt");

/// Advertised by the stub endpoint. Higher than any real release, so the check
/// reports an update no matter what version the crate is at.
const REMOTE_VERSION: &str = "9.9.9";

/// Manifest key this suite pins — the key a .deb install asks for. The plugin
/// derives `{os}-{arch}-{installer}` itself; the tests request it explicitly so
/// a change to that scheme fails here rather than silently passing against a
/// key nothing requests.
const TARGET: &str = "linux-x86_64-deb";

enum Payload {
    /// Exactly the bytes the fixture signature covers.
    Signed,
    /// One flipped byte, same signature — a corrupted download or a swapped
    /// asset on the release host.
    Tampered,
}

/// Serve `/latest.json` and the payload it points at, returning the manifest
/// URL. Binds port 0 so tests can run concurrently.
fn spawn_endpoint(kind: Payload) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub endpoint");
    let port = listener.local_addr().unwrap().port();

    let body = match kind {
        Payload::Signed => PAYLOAD.to_vec(),
        Payload::Tampered => {
            let mut bytes = PAYLOAD.to_vec();
            bytes[0] ^= 0xff;
            bytes
        }
    };
    let manifest = format!(
        concat!(
            r#"{{"version":"{version}","pub_date":"2026-01-01T00:00:00Z","platforms":"#,
            r#"{{"{target}":{{"signature":"{signature}","url":"http://127.0.0.1:{port}/payload"}}}}}}"#
        ),
        version = REMOTE_VERSION,
        target = TARGET,
        signature = SIGNATURE.trim(),
        port = port,
    );

    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = [0u8; 2048];
            let read = stream.read(&mut buf).unwrap_or(0);
            let wants_payload = String::from_utf8_lossy(&buf[..read]).contains("/payload");

            let (content_type, body) = if wants_payload {
                ("application/octet-stream", body.clone())
            } else {
                ("application/json", manifest.clone().into_bytes())
            };
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(&body);
        }
    });

    format!("http://127.0.0.1:{port}/latest.json")
}

/// Build a mock app whose updater config points at the stub. Plain HTTP needs
/// `dangerousInsecureTransportProtocol`; the shipped config uses HTTPS.
fn app_for(endpoint: &str) -> tauri::App<MockRuntime> {
    app_for_with_key(endpoint, PUBKEY)
}

fn app_for_with_key(endpoint: &str, pubkey: &str) -> tauri::App<MockRuntime> {
    let mut context = mock_context(noop_assets());
    context.config_mut().plugins.0.insert(
        "updater".into(),
        serde_json::json!({
            "endpoints": [endpoint],
            "pubkey": pubkey.trim(),
            "dangerousInsecureTransportProtocol": true,
        }),
    );

    mock_builder()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .build(context)
        .expect("failed to build updater test app")
}

fn check_and_download(kind: Payload) -> tauri_plugin_updater::Result<Vec<u8>> {
    let endpoint = spawn_endpoint(kind);
    let app = app_for(&endpoint);
    let updater = app
        .updater_builder()
        .target(TARGET)
        .build()
        .expect("updater config should be accepted");

    let update = tauri::async_runtime::block_on(updater.check())
        .expect("check should reach the endpoint")
        .expect("endpoint advertises a newer version");
    assert_eq!(update.version, REMOTE_VERSION);

    tauri::async_runtime::block_on(update.download(|_, _| {}, || {}))
}

#[test]
fn accepts_an_update_whose_payload_matches_its_signature() {
    let bytes = check_and_download(Payload::Signed).expect("correctly signed payload");
    assert_eq!(bytes, PAYLOAD);
}

#[test]
fn rejects_a_payload_that_does_not_match_its_signature() {
    let error = check_and_download(Payload::Tampered)
        .expect_err("a payload the signature does not cover must never be installed");
    let message = error.to_string().to_lowercase();
    assert!(
        message.contains("signature"),
        "expected a signature failure, got: {error}"
    );
}

#[test]
fn rejects_an_update_signed_by_a_different_key() {
    let endpoint = spawn_endpoint(Payload::Signed);
    let app = app_for_with_key(&endpoint, OTHER_PUBKEY);
    let updater = app
        .updater_builder()
        .target(TARGET)
        .build()
        .expect("updater config should be accepted");

    let update = tauri::async_runtime::block_on(updater.check())
        .expect("check should reach the endpoint")
        .expect("endpoint advertises a newer version");

    let error = tauri::async_runtime::block_on(update.download(|_, _| {}, || {}))
        .expect_err("a payload signed by another key must never be installed");
    let message = error.to_string().to_lowercase();
    assert!(
        message.contains("signature"),
        "expected a signature failure, got: {error}"
    );
}
