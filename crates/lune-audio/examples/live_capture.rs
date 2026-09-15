// SPDX-License-Identifier: GPL-3.0-or-later

//! Example: Live audio capture with sample count and buffer fill reporting.

use lune_audio::{AudioConfig, AudioStream};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = AudioConfig::default();
    let stream = AudioStream::new(config).expect("Failed to start audio stream");

    println!("Listening... (Ctrl+C to stop)\n");

    let mut chunk_count: u64 = 0;
    while let Some(chunk) = stream.next_chunk().await {
        chunk_count += 1;

        let fill = stream.buffer_fill();
        println!(
            "[{:6}] {} samples | buffer {:.0}%",
            chunk_count,
            chunk.samples.len(),
            fill * 100.0,
        );

        if fill > 0.8 {
            println!(
                "  WARNING: buffer {}% full - consumer may be slow",
                (fill * 100.0) as u32
            );
        }
    }
}
