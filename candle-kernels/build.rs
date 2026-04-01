use std::env;
use std::path::PathBuf;

const KERNEL_NAMES: &[&str] = &[
    "affine",
    "binary",
    "cast",
    "conv",
    "fill",
    "indexing",
    "quantized",
    "reduce",
    "sort",
    "ternary",
    "unary",
];

/// MOE kernel source files (relative to crate root).
const MOE_KERNEL_NAMES: &[&str] = &["moe_wmma", "moe_gguf", "moe_wmma_gguf"];

/// Which local headers (in order) should be prepended for runtime compilation.
/// System headers like <stdint.h>, <cmath>, <cuda_fp16.h> etc. are available via nvrtc
/// and do not need inlining. Only our custom .cuh files need it.
///
/// The dependency chain is:
///   compatibility.cuh  (includes cuda_fp16.h, cuda_bf16.h, cuda_fp8.h — system)
///   cuda_utils.cuh     (includes compatibility.cuh, <stdint.h>, <cmath>)
///   binary_op_macros.cuh (includes cuda_utils.cuh)
///
/// We inline all three in order for every kernel. This is redundant for kernels that
/// don't use all of them, but harmless — the unused definitions are just extra code
/// that nvrtc will parse but never emit.
const HEADERS_IN_ORDER: &[&str] = &[
    "compatibility.cuh",
    "cuda_utils.cuh",
    "binary_op_macros.cuh",
];

/// MOE kernels include different headers than the standard kernels.
/// Dependency chain:
///   gguf.cuh       (cuda_fp16.h, cuda_bf16.h — system; plus lots of quantization helpers)
///   moe_utils.cuh  (cuda.h, cuda_runtime.h — system; helper kernels + from_float)
///
/// moe_wmma.cu uses moe_utils.cuh
/// moe_gguf.cu uses gguf.cuh
/// moe_wmma_gguf.cu uses both gguf.cuh and moe_utils.cuh
const MOE_HEADERS_IN_ORDER: &[&str] = &["gguf.cuh", "moe_utils.cuh"];

/// Strip local #include "..." directives from source text.
fn strip_local_includes(source: &str) -> String {
    source
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with("#include") && trimmed.contains('"'))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Write a stripped version of a source file to `out_dir` and return the filename.
fn write_stripped(out_dir: &PathBuf, name: &str, suffix: &str, source: &str) -> String {
    let stripped = strip_local_includes(source);
    let fname = format!("{name}{suffix}");
    let path = out_dir.join(&fname);
    std::fs::write(&path, &stripped)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    fname
}

/// Generate a ptx.rs that embeds raw .cu source with headers inlined,
/// stripping local #include directives so nvrtc can compile them at runtime.
fn generate_source_embed(out_dir: &PathBuf) {
    use std::io::Write;

    let ptx_path = out_dir.join("ptx.rs");
    let mut f = std::fs::File::create(&ptx_path).expect("failed to create ptx.rs");

    // ---- Standard kernels ----
    for name in KERNEL_NAMES {
        let const_name = name.to_uppercase();

        let cu_path = format!("src/{name}.cu");
        let cu_source = std::fs::read_to_string(&cu_path)
            .unwrap_or_else(|e| panic!("failed to read {cu_path}: {e}"));
        write_stripped(out_dir, name, "_stripped.cu", &cu_source);

        for hdr in HEADERS_IN_ORDER {
            let hdr_path = format!("src/{hdr}");
            let hdr_source = std::fs::read_to_string(&hdr_path)
                .unwrap_or_else(|e| panic!("failed to read {hdr_path}: {e}"));
            let hdr_file = format!("{}_{}", name, hdr.replace('.', "_"));
            let hdr_stripped = strip_local_includes(&hdr_source);
            let hdr_stripped_path = out_dir.join(&hdr_file);
            std::fs::write(&hdr_stripped_path, &hdr_stripped).unwrap_or_else(|e| {
                panic!("failed to write {}: {e}", hdr_stripped_path.display())
            });
        }

        writeln!(f, "pub const {const_name}: &str = concat!(").unwrap();
        for hdr in HEADERS_IN_ORDER {
            let hdr_file = format!("{}_{}", name, hdr.replace('.', "_"));
            writeln!(
                f,
                "    include_str!(concat!(env!(\"OUT_DIR\"), \"/{hdr_file}\")), \"\\n\","
            )
            .unwrap();
        }
        writeln!(
            f,
            "    include_str!(concat!(env!(\"OUT_DIR\"), \"/{name}_stripped.cu\")),"
        )
        .unwrap();
        writeln!(f, ");").unwrap();
    }

    // ---- MOE kernels ----
    // Pre-process MOE headers once
    for hdr in MOE_HEADERS_IN_ORDER {
        let hdr_path = format!("src/moe/{hdr}");
        let hdr_source = std::fs::read_to_string(&hdr_path)
            .unwrap_or_else(|e| panic!("failed to read {hdr_path}: {e}"));
        let hdr_file = format!("moe_{}", hdr.replace('.', "_"));
        let hdr_stripped = strip_local_includes(&hdr_source);
        let hdr_stripped_path = out_dir.join(&hdr_file);
        std::fs::write(&hdr_stripped_path, &hdr_stripped).unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", hdr_stripped_path.display())
        });
    }

    for name in MOE_KERNEL_NAMES {
        let const_name = format!("MOE_{}", name.strip_prefix("moe_").unwrap_or(name))
            .to_uppercase();

        let cu_path = format!("src/moe/{name}.cu");
        let cu_source = std::fs::read_to_string(&cu_path)
            .unwrap_or_else(|e| panic!("failed to read {cu_path}: {e}"));
        write_stripped(out_dir, name, "_stripped.cu", &cu_source);

        writeln!(f, "pub const {const_name}: &str = concat!(").unwrap();
        for hdr in MOE_HEADERS_IN_ORDER {
            let hdr_file = format!("moe_{}", hdr.replace('.', "_"));
            writeln!(
                f,
                "    include_str!(concat!(env!(\"OUT_DIR\"), \"/{hdr_file}\")), \"\\n\","
            )
            .unwrap();
        }
        writeln!(
            f,
            "    include_str!(concat!(env!(\"OUT_DIR\"), \"/{name}_stripped.cu\")),"
        )
        .unwrap();
        writeln!(f, ");").unwrap();
    }
}

#[cfg(feature = "nvcc")]
fn try_nvcc_build(out_dir: &PathBuf) -> bool {
    // Build standard kernels to PTX
    let std_bindings = match cudaforge::KernelBuilder::new()
        .source_dir("src")
        .exclude(&["moe_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("candle-kernels: nvcc PTX build (standard) failed: {e}");
            return false;
        }
    };

    // Build MOE kernels to PTX (no longer a static lib — all host code is in Rust)
    let moe_bindings = match cudaforge::KernelBuilder::new()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("candle-kernels: nvcc PTX build (MOE) failed: {e}");
            return false;
        }
    };

    // Write standard bindings first, then append MOE bindings to the same file
    let ptx_path = out_dir.join("ptx.rs");

    if let Err(e) = std_bindings.write(&ptx_path) {
        eprintln!("candle-kernels: failed to write standard ptx.rs: {e}");
        return false;
    }

    // Append MOE bindings by writing to a temp file then concatenating
    let moe_ptx_path = out_dir.join("ptx_moe.rs");
    if let Err(e) = moe_bindings.write(&moe_ptx_path) {
        eprintln!("candle-kernels: failed to write MOE ptx.rs: {e}");
        return false;
    }
    // Append MOE content to the main ptx.rs
    let moe_content = std::fs::read_to_string(&moe_ptx_path)
        .expect("failed to read MOE ptx.rs");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(&ptx_path)
        .expect("failed to open ptx.rs for appending");
    write!(f, "\n{moe_content}").expect("failed to append MOE to ptx.rs");

    true
}

#[cfg(not(feature = "nvcc"))]
fn try_nvcc_build(_out_dir: &PathBuf) -> bool {
    false
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(candle_kernels_runtime_compile)");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    for name in KERNEL_NAMES {
        println!("cargo::rerun-if-changed=src/{name}.cu");
    }
    // MOE sources
    println!("cargo::rerun-if-changed=src/moe/gguf.cuh");
    println!("cargo::rerun-if-changed=src/moe/moe_utils.cuh");
    for name in MOE_KERNEL_NAMES {
        println!("cargo::rerun-if-changed=src/moe/{name}.cu");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if try_nvcc_build(&out_dir) {
        // nvcc succeeded — PTX is pre-compiled (both standard and MOE)
        eprintln!("candle-kernels: nvcc found, PTX pre-compiled (standard + MOE)");
    } else {
        // nvcc not available — embed raw .cu source for runtime compilation via nvrtc
        eprintln!("candle-kernels: nvcc not available, embedding .cu source for runtime compilation");
        println!("cargo:rustc-cfg=candle_kernels_runtime_compile");
        generate_source_embed(&out_dir);
    }
}
