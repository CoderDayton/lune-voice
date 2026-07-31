fn main() {
    // ggml's compute backends are built as dlopen-able plugins
    // (`GGML_BACKEND_DL`), which also makes libwhisper/libggml shared objects
    // rather than static archives. They ship next to the executable, so the
    // binary needs `$ORIGIN` on its run path or it cannot start outside the
    // build tree. `$ORIGIN` is expanded by the loader, so the literal dollar
    // sign has to survive both make and the shell: `-Wl,-z,origin` marks the
    // object as needing that expansion.
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,-z,origin");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
    }
    tauri_build::build();
}
