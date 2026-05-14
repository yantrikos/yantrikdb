// Build script for yantrikdb-python.
//
// **macOS-specific linker flags (closes #19).** pyo3's extension-module
// feature deliberately does NOT link against libpython; instead it
// expects symbols (PyGILState_Release, __Py_NoneStruct, ...) to be
// resolved by the host process at module load time. macOS's default
// linker rejects the resulting cdylib build with "symbol(s) not found
// for architecture arm64" because it can't see the Python runtime
// during link.
//
// maturin injects `-undefined dynamic_lookup` when it builds wheels via
// the pypi.yml workflow, but stock `cargo build --workspace` on macOS
// doesn't. This build.rs emits the same flag so stock cargo builds
// work on macOS — enables the CI test.yml workflow to drop
// `--exclude yantrikdb-python` from its `cargo build` step.
//
// `#[cfg(target_os = "macos")]` on a build script runs on the HOST,
// which is the platform doing the build. That's exactly what we want
// — the linker flags only need to be emitted when actually linking
// for macOS.
//
// No-op on Linux + Windows; pyo3's default linker behavior works there.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
