// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
//
// MOE kernels are now compiled at runtime via nvrtc (or pre-compiled to PTX by nvcc)
// and launched from Rust. No C FFI, no libmoe.a, no nvcc requirement.

#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

/// WMMA GEMM shared memory calculation constants (must match moe_wmma.cu)
#[cfg(feature = "cuda")]
const M_BLK: usize = 32;
#[cfg(feature = "cuda")]
const N_BLK: usize = 32;
#[cfg(feature = "cuda")]
const K_BLK: usize = 16;
#[cfg(feature = "cuda")]
const BLOCK_THREADS: u32 = 128;

/// GGUF decode constants
#[cfg(feature = "cuda")]
const MATRIX_ROW_PADDING: usize = 512;
#[cfg(feature = "cuda")]
const CUDA_QUANTIZE_BLOCK_SIZE: u32 = 256;
/// QK8_1 = 32 (from gguf.cuh)
#[cfg(feature = "cuda")]
const QK8_1: usize = 32;
/// sizeof(block_q8_1) = 2*sizeof(half) + 32 = 36 bytes
#[cfg(feature = "cuda")]
const SIZEOF_BLOCK_Q8_1: usize = 36;
#[cfg(feature = "cuda")]
const WARP_SIZE: u32 = 32;

#[cfg(feature = "cuda")]
fn pad(size: usize, padding: usize) -> usize {
    if padding == 0 {
        return size;
    }
    ((size + padding - 1) / padding) * padding
}

#[cfg(feature = "cuda")]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Launch the expert offset calculation kernels.
///
/// This replaces the C host functions `calculate_expert_offsets` and
/// `calculate_expert_offsets_light` from moe_utils.cuh.  Both paths now
/// use the single-block custom prefix-sum kernel (no thrust dependency).
#[cfg(feature = "cuda")]
fn launch_expert_offsets(
    dev: &candle::cuda_backend::CudaDevice,
    expert_ids: &cudarc::driver::CudaSlice<u32>,
    size_m: usize,
    expert_counts: &cudarc::driver::CudaSlice<u32>,
    expert_offsets: &cudarc::driver::CudaSlice<u32>,
    num_experts: usize,
) -> Result<()> {
    use candle::cuda_backend::kernels;
    use cudarc::driver::LaunchConfig;

    // Launch count_tokens_per_expert
    {
        let func = dev.get_or_load_func("count_tokens_per_expert", &kernels::MOE_WMMA)?;
        let threads = 256u32;
        let blocks = ceil_div(size_m, threads as usize) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let size_m_i32 = size_m as i32;
        let mut builder = func.builder();
        builder.arg(expert_ids);
        builder.arg(expert_counts);
        builder.arg(&size_m_i32);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                candle::Error::Msg(format!("count_tokens_per_expert launch: {e}"))
            })?;
        }
    }

    // Launch expert_prefix_sum
    {
        let func = dev.get_or_load_func("expert_prefix_sum", &kernels::MOE_WMMA)?;
        let mut scan_threads = num_experts;
        if scan_threads < 32 {
            scan_threads = 32;
        }
        let smem_size = (scan_threads * std::mem::size_of::<i32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (scan_threads as u32, 1, 1),
            shared_mem_bytes: smem_size,
        };
        let num_experts_i32 = num_experts as i32;
        let mut builder = func.builder();
        builder.arg(expert_counts);
        builder.arg(expert_offsets);
        builder.arg(&num_experts_i32);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                candle::Error::Msg(format!("expert_prefix_sum launch: {e}"))
            })?;
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::kernels;
    use candle::DType;
    use cudarc::driver::LaunchConfig;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        use candle::cuda_backend::kernels;
        use candle::cuda_backend::SlicePtrOrNull;
        use candle::DType;
        use cudarc::driver::LaunchConfig;

        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;

        let (input_storage, _) = input.storage_and_layout();
        let input_slice = match &*input_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights_storage, _) = weights.storage_and_layout();
        let weights_slice = match &*weights_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };

        let (sorted_token_ids_storage, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids_slice = match &*sorted_token_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids_storage, _) = experts_ids.storage_and_layout();
        let experts_ids_slice = match &*experts_ids_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_arg: SlicePtrOrNull<f32> = if let Some(tw) = &topk_weights {
            let (tw_storage, _) = tw.storage_and_layout();
            match &*tw_storage {
                candle::Storage::Cuda(c) => {
                    SlicePtrOrNull::Ptr(c.as_cuda_slice::<f32>()?.clone())
                }
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            }
        } else {
            SlicePtrOrNull::Null
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
        // expert_counts must be zero-initialized for atomicAdd
        let expert_counts = dev.alloc_zeros::<u32>(num_experts)?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        // Calculate expert offsets
        launch_expert_offsets(
            dev,
            &experts_ids_slice,
            size_m,
            &expert_counts,
            &expert_offsets,
            num_experts,
        )?;

        // Calculate grid/block dims
        let grid_n = ceil_div(size_n, N_BLK) as u32;

        // Shared memory
        let a_sh_bytes = M_BLK * K_BLK * 2; // sizeof(half) = 2
        let b_sh_bytes = N_BLK * K_BLK * 2;
        let c_sh_bytes = M_BLK * N_BLK * 4; // sizeof(float) = 4
        let ab_bytes = a_sh_bytes + b_sh_bytes;
        let pad_amount = (16 - (ab_bytes % 16)) % 16;
        let smem_bytes = (ab_bytes + pad_amount + c_sh_bytes) as u32;

        // Select kernel by dtype and mode
        let kernel_name = match (input.dtype(), is_prefill) {
            (DType::F16, true) => "moe_wmma_half_16_16_2",
            (DType::F16, false) => "moe_wmma_half_8_32_1",
            (DType::BF16, true) => "moe_wmma_bf16_16_16_2",
            (DType::BF16, false) => "moe_wmma_bf16_8_32_1",
            _ => candle::bail!("moe_gemm_wmma only accepts f16/bf16 inputs"),
        };

        let func = dev.get_or_load_func(kernel_name, &kernels::MOE_WMMA)?;
        let cfg = LaunchConfig {
            grid_dim: (num_experts as u32, grid_n, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let num_experts_i32 = num_experts as i32;
        let topk_i32 = topk as i32;
        let size_m_i32 = size_m as i32;
        let size_n_i32 = size_n as i32;
        let size_k_i32 = size_k as i32;
        let mut builder = func.builder();
        // Kernel args: input, weights, sorted_token_ids, expert_offsets, topk_weights, output,
        //              num_experts, topk, size_m, size_n, size_k
        builder.arg(input_slice);
        builder.arg(weights_slice);
        builder.arg(sorted_token_ids_slice);
        builder.arg(&expert_offsets);
        topk_weights_arg.builder_arg(&mut builder);
        builder.arg(&output);
        builder.arg(&num_experts_i32);
        builder.arg(&topk_i32);
        builder.arg(&size_m_i32);
        builder.arg(&size_n_i32);
        builder.arg(&size_k_i32);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| candle::Error::Msg(format!("moe_wmma launch: {e}")))?;
        }

        use candle::op::BackpropOp;
        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        candle::DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        candle::DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle::bail!("moe_gemm only accepts f16/bf16 inputs")
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!("moe_gemm requires the `cuda` feature")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    use candle::cuda_backend::kernels;
    use candle::cuda_backend::SlicePtrOrNull;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use cudarc::driver::{DevicePtr, LaunchConfig};
    use half::{bf16, f16};

    let (mut size_m, size_k) = input.dims2()?;
    if topk_weights.is_none() {
        size_m *= topk;
    }
    let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
    assert!(
        size_k == size_k1,
        "input {:?} and weight {:?} last dim mismatch!",
        size_k,
        size_k1,
    );
    let dev = input.device().as_cuda_device()?;

    // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3, Q5K: 4, Q6K: 5
    let gguf_dtype: i32 = match weights.dtype() {
        GgmlDType::Q8_0 => 0,
        GgmlDType::Q4K => 1,
        GgmlDType::Q2K => 2,
        GgmlDType::Q3K => 3,
        GgmlDType::Q5K => 4,
        GgmlDType::Q6K => 5,
        _ => {
            candle::bail!(
                "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
            )
        }
    };

    // weight_ptr is a raw device address
    let weight_ptr = weights.device_ptr()? as u64;

    let topk_weights_arg: SlicePtrOrNull<f32> = if let Some(tw) = &topk_weights {
        let (tw_storage, _) = tw.storage_and_layout();
        match &*tw_storage {
            candle::Storage::Cuda(c) => {
                SlicePtrOrNull::Ptr(c.as_cuda_slice::<f32>()?.clone())
            }
            _ => candle::bail!("topk_weights must be a cuda tensor"),
        }
    } else {
        SlicePtrOrNull::Null
    };

    let (sorted_token_ids_storage, _) = sorted_token_ids.storage_and_layout();
    let sorted_token_ids_slice = match &*sorted_token_ids_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
    };
    let (experts_ids_storage, _) = experts_ids.storage_and_layout();
    let experts_ids_slice = match &*experts_ids_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle::bail!("experts_ids must be a cuda tensor"),
    };

    let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;

    assert!(size_k % 8 == 0, "size_k must divisible by 8");

    let num_experts_i32 = num_experts as i32;
    let topk_i32 = topk as i32;
    let size_m_i32 = size_m as i32;
    let size_n_i32 = size_n as i32;
    let size_k_i32 = size_k as i32;

    if is_prefill {
        // ---- Prefill path: uses WMMA + GGUF dequant ----
        let expert_counts = dev.alloc_zeros::<u32>(num_experts)?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        launch_expert_offsets(
            dev,
            &experts_ids_slice,
            size_m,
            &expert_counts,
            &expert_offsets,
            num_experts,
        )?;

        let grid_n = ceil_div(size_n, N_BLK) as u32;

        // Determine qk and block_size_bytes for shared memory calculation
        let (qk, block_size_bytes): (usize, usize) = match gguf_dtype {
            0 => (32, 34),   // QK8_0=32, sizeof(block_q8_0) = 2 + 32 = 34
            1 => (256, 144), // QK_K=256, sizeof(block_q4_K)
            2 => (256, 84),  // sizeof(block_q2_K)
            3 => (256, 110), // sizeof(block_q3_K)
            4 => (256, 176), // sizeof(block_q5_K)
            5 => (256, 210), // sizeof(block_q6_K)
            _ => candle::bail!("unsupported gguf_dtype"),
        };

        let a_sh_bytes = M_BLK * qk * 2;
        let b_sh_bytes = N_BLK * qk * 2;
        let b_quant_sh_bytes = N_BLK * block_size_bytes;
        let c_sh_bytes = M_BLK * N_BLK * 4;
        let mut smem_total = a_sh_bytes + b_sh_bytes + b_quant_sh_bytes;
        let c_offset = smem_total % 4;
        if c_offset != 0 {
            smem_total += 4 - c_offset;
        }
        smem_total += c_sh_bytes;

        // Determine block dims and kernel name based on dtype and gguf_type
        let (wrap_size, kernel_name): (u32, &str) = match (dtype, gguf_dtype) {
            (DType::F16, 0) => (32, "moe_gguf_prefill_half_q8_0"),
            (DType::F16, 1) => (32, "moe_gguf_prefill_half_q4k"),
            (DType::F16, 2) => (64, "moe_gguf_prefill_half_q2k"),
            (DType::F16, 3) => (64, "moe_gguf_prefill_half_q3k"),
            (DType::F16, 4) => (64, "moe_gguf_prefill_half_q5k"),
            (DType::F16, 5) => (64, "moe_gguf_prefill_half_q6k"),
            (DType::BF16, 0) => (32, "moe_gguf_prefill_bf16_q8_0"),
            (DType::BF16, 1) => (32, "moe_gguf_prefill_bf16_q4k"),
            (DType::BF16, 2) => (64, "moe_gguf_prefill_bf16_q2k"),
            (DType::BF16, 3) => (64, "moe_gguf_prefill_bf16_q3k"),
            (DType::BF16, 4) => (64, "moe_gguf_prefill_bf16_q5k"),
            (DType::BF16, 5) => (64, "moe_gguf_prefill_bf16_q6k"),
            _ => candle::bail!("unsupported dtype/gguf combination for prefill"),
        };

        let cfg = LaunchConfig {
            grid_dim: (num_experts as u32, grid_n, 1),
            block_dim: (wrap_size, 4, 1), // WARPS_PER_BLOCK = 4
            shared_mem_bytes: smem_total as u32,
        };

        // Convert input to the target dtype
        let input_conv = input.to_dtype(dtype)?;
        let (input_conv_storage, _) = input_conv.storage_and_layout();
        let input_ptr: u64 = match &*input_conv_storage {
            candle::Storage::Cuda(c) => {
                if dtype == DType::F16 {
                    let s = c.as_cuda_slice::<f16>()?;
                    s.device_ptr(s.stream()).0
                } else {
                    let s = c.as_cuda_slice::<bf16>()?;
                    s.device_ptr(s.stream()).0
                }
            }
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let func = dev.get_or_load_func(kernel_name, &kernels::MOE_WMMA_GGUF)?;
        let mut builder = func.builder();
        // Args: input(void*), weights(uint8_t*), sorted_token_ids, expert_offsets,
        //       topk_weights, output, num_experts, topk, size_m, size_n, size_k, gguf_dtype
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(sorted_token_ids_slice);
        builder.arg(&expert_offsets);
        topk_weights_arg.builder_arg(&mut builder);
        builder.arg(&output);
        builder.arg(&num_experts_i32);
        builder.arg(&topk_i32);
        builder.arg(&size_m_i32);
        builder.arg(&size_n_i32);
        builder.arg(&size_k_i32);
        builder.arg(&gguf_dtype);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| candle::Error::Msg(format!("moe_gguf_prefill launch: {e}")))?;
        }
    } else {
        // ---- Decode path: quantize input to q8_1, then dot product ----
        let (input_storage, _) = input.storage_and_layout();
        let input_f32 = match &*input_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let kx_padded = pad(size_k, MATRIX_ROW_PADDING);
        let m = if topk_weights.is_some() {
            size_m
        } else {
            size_m / topk
        };

        // Allocate temp buffer for quantized input (as raw bytes via u8 slice)
        let y_size_in_bytes = m * (kx_padded / QK8_1) * SIZEOF_BLOCK_Q8_1;
        let y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes) }?;

        // Launch quantize_q8_1
        {
            let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE as usize) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, m as u32, 1),
                block_dim: (CUDA_QUANTIZE_BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };
            let kx_i32 = size_k as i32;
            let kx_padded_i32 = kx_padded as i32;

            let func = dev.get_or_load_func("moe_quantize_q8_1", &kernels::MOE_GGUF)?;
            let mut builder = func.builder();
            // Args: x(float*), vy(void*), kx, kx_padded
            builder.arg(input_f32);
            builder.arg(&y_q8_1);
            builder.arg(&kx_i32);
            builder.arg(&kx_padded_i32);
            unsafe {
                builder.launch(cfg).map_err(|e| {
                    candle::Error::Msg(format!("quantize_q8_1 launch: {e}"))
                })?;
            }
        }

        // Launch main GGUF kernel
        {
            let n_wraps = 4u32;
            let cfg = LaunchConfig {
                grid_dim: (ceil_div(size_n, n_wraps as usize) as u32, size_m as u32, 1),
                block_dim: (WARP_SIZE, n_wraps, 1),
                shared_mem_bytes: {
                    let qk: usize = match gguf_dtype {
                        0 => 32,
                        _ => 256,
                    };
                    let block_q_size: usize = match gguf_dtype {
                        0 => 34,
                        1 => 144,
                        2 => 84,
                        3 => 110,
                        4 => 176,
                        5 => 210,
                        _ => unreachable!(),
                    };
                    (size_k / qk * block_q_size * n_wraps as usize + 1024) as u32
                },
            };

            let kernel_name = match gguf_dtype {
                0 => "moe_gguf_q8_0",
                1 => "moe_gguf_q4k",
                2 => "moe_gguf_q2k",
                3 => "moe_gguf_q3k",
                4 => "moe_gguf_q5k",
                5 => "moe_gguf_q6k",
                _ => candle::bail!("unsupported gguf_dtype"),
            };

            let kx_padded_i32 = kx_padded as i32;

            let func = dev.get_or_load_func(kernel_name, &kernels::MOE_GGUF)?;
            let mut builder = func.builder();
            // Args: weights(void*), inputs(void*=quantized), sorted_token_ids, expert_ids,
            //       topk_weights, outputs, num_experts, topk,
            //       size_m, size_n, size_k, k_padded
            builder.arg(&weight_ptr);
            builder.arg(&y_q8_1);
            builder.arg(sorted_token_ids_slice);
            builder.arg(experts_ids_slice);
            topk_weights_arg.builder_arg(&mut builder);
            builder.arg(&output);
            builder.arg(&num_experts_i32);
            builder.arg(&topk_i32);
            builder.arg(&size_m_i32);
            builder.arg(&size_n_i32);
            builder.arg(&size_k_i32);
            builder.arg(&kx_padded_i32);
            unsafe {
                builder.launch(cfg).map_err(|e| {
                    candle::Error::Msg(format!("moe_gguf kernel launch: {e}"))
                })?;
            }
        }
        // y_q8_1 is dropped here, freeing the temp buffer
    }

    use candle::op::BackpropOp;
    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let output = Tensor::from_storage(
        candle::Storage::Cuda(output),
        (size_m, size_n),
        BackpropOp::none(),
        false,
    );

    Ok(output)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle::DType,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf requires the `cuda` feature")
}
