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

/// Generate a ptx.rs that embeds raw .cu source with headers inlined,
/// stripping local #include directives so nvrtc can compile them at runtime.
fn generate_source_embed(out_dir: &PathBuf) {
    use std::io::Write;

    let ptx_path = out_dir.join("ptx.rs");
    let mut f = std::fs::File::create(&ptx_path).expect("failed to create ptx.rs");

    for name in KERNEL_NAMES {
        let const_name = name.to_uppercase();

        // Read the .cu source and strip local includes (#include "...")
        let cu_path = format!("src/{name}.cu");
        let cu_source = std::fs::read_to_string(&cu_path)
            .unwrap_or_else(|e| panic!("failed to read {cu_path}: {e}"));
        let stripped: String = cu_source
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !(trimmed.starts_with("#include") && trimmed.contains('"'))
            })
            .collect::<Vec<_>>()
            .join("\n");

        let stripped_path = out_dir.join(format!("{name}_stripped.cu"));
        std::fs::write(&stripped_path, &stripped)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", stripped_path.display()));

        // Process headers: strip their local includes too (they're all included in order)
        for hdr in HEADERS_IN_ORDER {
            let hdr_path = format!("src/{hdr}");
            let hdr_source = std::fs::read_to_string(&hdr_path)
                .unwrap_or_else(|e| panic!("failed to read {hdr_path}: {e}"));
            let hdr_stripped: String = hdr_source
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    !(trimmed.starts_with("#include") && trimmed.contains('"'))
                })
                .collect::<Vec<_>>()
                .join("\n");
            let hdr_stripped_path = out_dir.join(format!("{}_{}", name, hdr.replace('.', "_")));
            std::fs::write(&hdr_stripped_path, &hdr_stripped).unwrap_or_else(|e| {
                panic!("failed to write {}: {e}", hdr_stripped_path.display())
            });
        }

        // Write the const definition using concat! + include_str!
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
}

#[cfg(feature = "nvcc")]
fn try_nvcc_build(out_dir: &PathBuf) -> bool {
    let bindings = match cudaforge::KernelBuilder::new()
        .source_dir("src")
        .exclude(&["moe_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("candle-kernels: nvcc PTX build failed: {e}");
            return false;
        }
    };

    let ptx_path = out_dir.join("ptx.rs");
    if let Err(e) = bindings.write(&ptx_path) {
        eprintln!("candle-kernels: failed to write ptx.rs: {e}");
        return false;
    }

    true
}

#[cfg(not(feature = "nvcc"))]
fn try_nvcc_build(_out_dir: &PathBuf) -> bool {
    false
}

#[cfg(feature = "moe")]
fn build_moe(out_dir: &PathBuf) {
    let mut moe_builder = cudaforge::KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder
        .build_lib(out_dir.join("libmoe.a"))
        .expect("MOE kernel build failed (moe feature requires nvcc)");
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
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

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if try_nvcc_build(&out_dir) {
        // nvcc succeeded — PTX is pre-compiled
        eprintln!("candle-kernels: nvcc found, PTX pre-compiled");

        #[cfg(feature = "moe")]
        build_moe(&out_dir);
    } else {
        // nvcc not available — embed raw .cu source for runtime compilation via nvrtc
        eprintln!("candle-kernels: nvcc not available, embedding .cu source for runtime compilation");
        println!("cargo:rustc-cfg=candle_kernels_runtime_compile");
        generate_source_embed(&out_dir);
    }
}
