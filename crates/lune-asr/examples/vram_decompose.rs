//! Per-session VRAM decomposition: loads each ASR session one at a time and
//! prints the process's GPU memory after each, so model-storage cost can be
//! separated from CUDA runtime overhead.
//!
//! Run: cargo run --release --features cuda -p lune-asr --example vram_decompose

use lune_asr::ort_utils::OrtModelLoader;

fn vram_mib() -> u64 {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut parts = l.split(',').map(str::trim);
            let p = parts.next()?;
            let mem = parts.next()?.parse::<u64>().ok()?;
            (p == pid).then_some(mem)
        })
        .sum()
}

fn main() {
    let _ = ort::init().commit();
    let home = std::env::var("HOME").expect("HOME");
    let dir = std::path::PathBuf::from(home).join(".local/share/lune-voice/models/qwen3-asr-0.6b");
    let loader = OrtModelLoader::new(dir, true, true);

    println!("baseline (no sessions):        {:>6} MiB", vram_mib());

    let _embed = loader
        .load_first(&["qwen3-asr-embed-tokens.onnx"], "embed")
        .expect("embed");
    let after_embed = vram_mib();
    println!("+ embed (311 MB f16 on disk):  {after_embed:>6} MiB");

    let _encoder = loader
        .load_first(&["qwen3-asr-encoder.onnx"], "encoder")
        .expect("encoder");
    let after_enc = vram_mib();
    println!("+ encoder (378 MB f16 on disk):{after_enc:>6} MiB");

    let _decoder = loader
        .load_first(&["qwen3-asr-decoder.onnx"], "decoder")
        .expect("decoder");
    let after_dec = vram_mib();
    println!("+ decoder (596 MB int8 on disk):{after_dec:>5} MiB");
}
