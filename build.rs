use std::env;
use std::process::Command;

fn main() {
    // Tell Rust about our custom cfg flags
    println!("cargo::rustc-check-cfg=cfg(gpu_cuda_available)");
    println!("cargo::rustc-check-cfg=cfg(gpu_metal_available)");
    println!("cargo::rustc-check-cfg=cfg(target_platform_macos)");
    println!("cargo::rustc-check-cfg=cfg(target_platform_linux)");
    println!("cargo::rustc-check-cfg=cfg(target_platform_windows)");
    println!("cargo::rustc-check-cfg=cfg(target_platform_other)");
    println!("cargo::rustc-check-cfg=cfg(gpu_acceleration_available)");
    println!("cargo::rustc-check-cfg=cfg(cuda_compute_available)");
    println!("cargo::rustc-check-cfg=cfg(metal_compute_available)");
    println!("cargo::rustc-check-cfg=cfg(simd_support)");
    println!("cargo::rustc-check-cfg=cfg(cpu_optimizations)");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=LOKI_FEATURES");

    // Detect target platform and set appropriate features
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // Comprehensive platform detection
    let platform_info = detect_platform_capabilities(&target_os, &target_arch);
    configure_build(&platform_info);
    print_build_recommendations(&platform_info);
    check_feature_compatibility(&platform_info);
}

#[derive(Debug)]
struct PlatformCapabilities {
    os: String,
    arch: String,
    has_cuda: bool,
    cuda_version: Option<String>,
    has_metal: bool,
    has_simd: bool,
    recommended_features: String,
}

fn detect_platform_capabilities(target_os: &str, target_arch: &str) -> PlatformCapabilities {
    // Check CUDA availability on ALL platforms, not just Linux
    let (has_cuda, cuda_version) = check_cuda_detailed();

    // Metal is available on macOS
    let has_metal = target_os == "macos";

    // SIMD support is available on most modern architectures
    let has_simd = matches!(target_arch, "x86_64" | "aarch64");

    let recommended_features = determine_recommended_features(target_os, has_cuda, has_metal);

    PlatformCapabilities {
        os: target_os.to_string(),
        arch: target_arch.to_string(),
        has_cuda,
        cuda_version,
        has_metal,
        has_simd,
        recommended_features,
    }
}

fn configure_build(info: &PlatformCapabilities) {
    // Set platform-specific configuration flags
    match info.os.as_str() {
        "macos" => {
            println!("cargo:rustc-cfg=target_platform_macos");
            if info.has_metal {
                println!("cargo:rustc-cfg=gpu_metal_available");
            }
            if info.has_cuda {
                println!("cargo:rustc-cfg=gpu_cuda_available");
                println!("cargo:warning=CUDA detected on macOS - experimental support enabled");
            }
        }
        "linux" => {
            println!("cargo:rustc-cfg=target_platform_linux");
            if info.has_cuda {
                println!("cargo:rustc-cfg=gpu_cuda_available");
                if let Some(ref version) = info.cuda_version {
                    println!("cargo:rustc-env=CUDA_VERSION={version}");
                }
            }
        }
        "windows" => {
            println!("cargo:rustc-cfg=target_platform_windows");
            if info.has_cuda {
                println!("cargo:rustc-cfg=gpu_cuda_available");
                if let Some(ref version) = info.cuda_version {
                    println!("cargo:rustc-env=CUDA_VERSION={version}");
                }
            }
        }
        _ => {
            println!("cargo:rustc-cfg=target_platform_other");
            if info.has_cuda {
                println!("cargo:rustc-cfg=gpu_cuda_available");
                println!(
                    "cargo:warning=CUDA detected on {} - experimental support enabled",
                    info.os
                );
            }
        }
    }

    // Configure GPU acceleration flags based on actual availability
    if info.has_cuda {
        println!("cargo:rustc-cfg=gpu_acceleration_available");
        println!("cargo:rustc-cfg=cuda_compute_available");
    }

    if info.has_metal {
        println!("cargo:rustc-cfg=gpu_acceleration_available");
        println!("cargo:rustc-cfg=metal_compute_available");
    }

    // Configure SIMD support
    if info.has_simd {
        println!("cargo:rustc-cfg=simd_support");
        println!("cargo:rustc-cfg=cpu_optimizations");
    }

    // Set recommended features environment variable
    println!("cargo:rustc-env=LOKI_RECOMMENDED_FEATURES={}", info.recommended_features);
}

fn print_build_recommendations(info: &PlatformCapabilities) {
    println!("cargo:warning=");
    println!("cargo:warning=🚀 Loki AI Build Configuration:");
    let platform = format!("{} {}", info.os, info.arch);
    println!("cargo:warning=  Platform: {platform}");

    let cuda_status = if info.has_cuda {
        format!("Available ({})", info.cuda_version.as_deref().unwrap_or("unknown version"))
    } else {
        "Not available".to_string()
    };
    println!("cargo:warning=  CUDA: {cuda_status}");

    let metal_status = if info.has_metal { "Available" } else { "Not available" };
    println!("cargo:warning=  Metal: {metal_status}");

    let simd_status = if info.has_simd { "Supported" } else { "Not supported" };
    println!("cargo:warning=  SIMD: {simd_status}");

    // Determine the best GPU backend for this platform
    let gpu_backend = if info.has_cuda && info.has_metal {
        "Both CUDA and Metal available - runtime selection enabled"
    } else if info.has_cuda {
        "CUDA GPU acceleration available"
    } else if info.has_metal {
        "Metal GPU acceleration available"
    } else {
        "CPU-only mode (no GPU acceleration detected)"
    };
    println!("cargo:warning=  GPU Backend: {gpu_backend}");

    println!("cargo:warning=");
    println!("cargo:warning=💡 Recommended build command:");
    let recommended = &info.recommended_features;
    println!("cargo:warning=  cargo build --features={recommended}");
    println!("cargo:warning=");
}

fn check_feature_compatibility(info: &PlatformCapabilities) {
    // Check for potentially problematic feature combinations
    let enabled_features = get_enabled_features();

    // Only warn if CUDA is enabled but not available
    if enabled_features.iter().any(|f| f == "cuda") && !info.has_cuda {
        println!("cargo:warning=⚠️  WARNING: CUDA features enabled but CUDA toolkit not detected");
        println!("cargo:warning=   Build will continue with fallback backend");
    }

    // Only warn if Metal is enabled but not available
    if enabled_features.iter().any(|f| f == "metal") && !info.has_metal {
        println!("cargo:warning=⚠️  WARNING: Metal features enabled but Metal not available");
        println!("cargo:warning=   Build will continue with fallback backend");
    }

    // Provide optimization suggestions
    if !enabled_features.iter().any(|f| f.contains("all-") || f == "simd-optimizations") {
        println!("cargo:warning=💡 Consider using optimized feature sets:");
        println!("cargo:warning=   --features={}", info.recommended_features);
    }
}

fn get_enabled_features() -> Vec<String> {
    env::vars()
        .filter_map(|(key, _)| {
            if key.starts_with("CARGO_FEATURE_") {
                Some(key.strip_prefix("CARGO_FEATURE_")?.to_lowercase().replace('_', "-"))
            } else {
                None
            }
        })
        .collect()
}

fn determine_recommended_features(os: &str, has_cuda: bool, has_metal: bool) -> String {
    match (os, has_cuda, has_metal) {
        ("linux", true, _) => "all-linux",
        ("macos", _, true) => "all-macos",
        ("macos", true, false) => "all-macos", // CUDA on macOS
        ("windows", true, _) => "all-windows", // Windows with CUDA
        (_, true, _) => "all-safe,cuda",       // Other platforms with CUDA
        _ => "all-safe",
    }
    .to_string()
}

fn check_cuda_detailed() -> (bool, Option<String>) {
    // First check with nvcc
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            let version_output = String::from_utf8_lossy(&output.stdout);
            // Extract version from CUDA compiler output
            for line in version_output.lines() {
                if line.contains("release") {
                    if let Some(version_part) = line.split("release ").nth(1) {
                        if let Some(version) = version_part.split(',').next() {
                            return (true, Some(version.trim().to_string()));
                        }
                    }
                }
            }
            return (true, Some("unknown".to_string()));
        }
    }

    // Fallback to the original CUDA detection
    (is_cuda_available(), None)
}

fn is_cuda_available() -> bool {
    // Check for CUDA toolkit installation
    if which::which("nvcc").is_ok() {
        return true;
    }

    // Check common CUDA installation paths across platforms
    let cuda_paths = match env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        "windows" => vec![
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\CUDA",
        ],
        "macos" => vec!["/usr/local/cuda", "/opt/cuda", "/Developer/NVIDIA/CUDA-*"],
        _ => vec!["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda", "/usr/cuda"],
    };

    for path in &cuda_paths {
        if path.contains('*') {
            // Handle glob patterns for version-specific directories
            if let Ok(entries) = std::fs::read_dir(path.trim_end_matches("*")) {
                for entry in entries.flatten() {
                    if entry.path().join("bin/nvcc").exists() {
                        return true;
                    }
                }
            }
        } else if std::path::Path::new(&format!("{path}/bin/nvcc")).exists() {
            return true;
        }
    }

    false
}
