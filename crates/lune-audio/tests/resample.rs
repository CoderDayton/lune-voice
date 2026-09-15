// SPDX-License-Identifier: GPL-3.0-or-later

//! Guards the capture resampler across rubato upgrades: the path only runs
//! inside the cpal callback, so nothing else exercises it.

use audioadapter_buffers::direct::SequentialSlice;
use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};

#[test]
fn resamples_48k_sine_to_16k_at_the_same_frequency() {
    let native = 48_000usize;
    let target = 16_000usize;
    let chunk = 480usize;
    let mut r = Async::<f32>::new_poly(
        target as f64 / native as f64,
        1.2,
        PolynomialDegree::Septic,
        chunk,
        1,
        FixedAsync::Input,
    )
    .expect("construct resampler");

    let freq = 440.0f64;
    let mut out: Vec<f32> = Vec::new();
    for block in 0..100 {
        let input: Vec<f32> = (0..chunk)
            .map(|i| {
                let n = (block * chunk + i) as f64;
                (2.0 * std::f64::consts::PI * freq * n / native as f64).sin() as f32
            })
            .collect();
        let adapter = SequentialSlice::new(input.as_slice(), 1, chunk).expect("adapter");
        let produced = r.process(&adapter, None).expect("process");
        out.extend_from_slice(&produced.take_data());
    }

    // Roughly 1/3 of the input frames come back out.
    let expected = 100 * chunk / 3;
    let ratio = out.len() as f64 / expected as f64;
    assert!(
        (0.95..1.05).contains(&ratio),
        "output length {} not ~{expected} (ratio {ratio})",
        out.len()
    );

    // Amplitude preserved: a unit sine stays a unit sine.
    let peak = out.iter().skip(200).fold(0.0f32, |a, s| a.max(s.abs()));
    assert!((0.9..1.1).contains(&peak), "peak amplitude {peak} not ~1.0");

    // Frequency preserved: count zero crossings over the settled region.
    let settled = &out[200..out.len() - 200];
    let crossings = settled
        .windows(2)
        .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
        .count();
    let seconds = settled.len() as f64 / target as f64;
    let measured = crossings as f64 / seconds;
    assert!(
        (measured - freq).abs() < 5.0,
        "measured {measured} Hz, expected {freq} Hz"
    );
}
