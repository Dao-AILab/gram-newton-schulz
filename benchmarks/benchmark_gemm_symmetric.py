import argparse
from typing import Tuple, Type, Optional
import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass import Int32
from quack.gemm_sm90 import GemmSm90, TileSchedulerOptions
from quack.gemm_symmetric import GemmSymmetricSm90, GemmSymmetricSm100
from quack.varlen_utils import VarlenArguments
from quack.cute_dsl_utils import get_device_capacity

def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL symmetric GEMM on Hopper.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--d_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=None,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--d_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations to run the kernel",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    return args


def run(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    c_dtype: Optional[Type[cutlass.Numeric]],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    d_major: str,
    c_major: str,
    warmup_iterations: int,
    iterations: int,
    **kwargs,
):
    """
    Prepare A/B/D/C tensors, launch GPU kernel, and benchmark timing.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param a_dtype: Data type for input tensor A
    :type a_dtype: Type[cutlass.Numeric]
    :param b_dtype: Data type for input tensor B
    :type b_dtype: Type[cutlass.Numeric]
    :param d_dtype: Data type for output tensor C
    :type d_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/d_major: Memory layout of tensor A/B/C
    :type a_major/b_major/d_major: str
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    """

    device_capacity = get_device_capacity()
    if isinstance(device_capacity, tuple):
        device_capacity = device_capacity[0]

    tile_m = 256 if device_capacity >= 10 else 128
    tile_shape_mn = (tile_m, 256)
    cluster_shape_mn = (2, 1)
    persistent = True
    dynamic_persistent = False
    pingpong = False
    varlen_m = False
    varlen_k = False
    gather_A = False
    fp8_fast_accum = False
    mCuSeqlensM = None
    mCuSeqlensK = None
    mAIdx = None

    if device_capacity == 12:
        arch_name = "RTX 5090 (SM120)"
    elif device_capacity == 10:
        arch_name = "Blackwell (SM100)"
    elif device_capacity == 9:
        arch_name = "Hopper (SM90)"
    else:
        arch_name = f"SM{device_capacity}0"

    print(f"Running {arch_name} Dense GEMM with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, D dtype: {d_dtype}, C_dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, D: {d_major}")
    print(f"Tile Shape: {tile_shape_mn}, Cluster Shape: {cluster_shape_mn}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")

    m, n, k, l = mnkl
    cluster_shape_mnk = (*cluster_shape_mn, 1)

    if not GemmSm90.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, d_dtype, a_major, b_major
    ):
        raise TypeError(
            f"Skipping due to unsupported combination of types and majors: {a_dtype}, {b_dtype}, {acc_dtype}, {d_dtype}, {a_major=}, {b_major=}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True):
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        torch_dtype = cutlass_torch.dtype(dtype)
        gen_dtype = (
            torch_dtype
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.bfloat16
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            gen_dtype,
            permute_order=permute_order,
            # init_type=cutlass.torch.TensorInitType.RANDOM,
            # init_config=cutlass.torch.RandomInitConfig(
            #     min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            # ),
            init_type=cutlass.torch.TensorInitType.GAUSSIAN,
            init_config=cutlass.torch.GaussianInitConfig(std=k ** (-0.5), scale=1),
        ).to(torch_dtype)
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        torch_tensor_view = (
            torch_tensor
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch_tensor.view(torch.uint8)
        )
        cute_tensor = from_dlpack(torch_tensor_view, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=(0 if is_mode0_major else 1))
            cute_tensor = cute_tensor.mark_compact_shape_dynamic(
                mode=(1 if not is_mode0_major else 0),
                stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
                divisibility=(128 // dtype.width),
            )
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    b, mB, b_torch = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
    _, mD, d_torch = create_and_permute_tensor(l, m, n, d_major == "m", d_dtype)
    if c_dtype is not None:
        c, mC, c_torch = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
    else:
        c, mC, c_torch = None, None, None
   
    if device_capacity == 10:  # Blackwell (SM100)
        gemm = GemmSymmetricSm100(
            acc_dtype,
            a_dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
    elif device_capacity == 9:  # Hopper (SM90)
        gemm = GemmSymmetricSm90(
            acc_dtype,
            a_dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            pingpong=pingpong,
            is_persistent=persistent,
            fp8_fast_accum=fp8_fast_accum,
            gather_A=gather_A,
        )
    else:
        raise ValueError(f"Unsupported GPU architecture: SM{device_capacity}0. Only Hopper (SM90) and Blackwell (SM100) are supported.")

    # Compute max active clusters on current device
    if persistent:
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )
        if dynamic_persistent:
            tile_count_semaphore = torch.zeros(1, dtype=torch.int32, device="cuda")
        else:
            tile_count_semaphore = None
        # max_active_clusters = 1
    else:
        max_active_clusters = 0
        tile_count_semaphore = None
    batch_idx_permute_tensor = None
    scheduler_args = TileSchedulerOptions(
        Int32(max_active_clusters),
        tile_count_semaphore=make_ptr(
            Int32, tile_count_semaphore.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ) if tile_count_semaphore is not None else None,
        batch_idx_permute=batch_idx_permute_tensor
    )

    if d_torch is not None:
        mPostAct = from_dlpack(d_torch, assumed_align=16)
        mPostAct.element_type = d_dtype
        mPostAct = mPostAct.mark_layout_dynamic(leading_dim=(0 if d_major == "m" else 1))
    else:
        mPostAct = None
    activation = None 
    epi_args = gemm.EpilogueArguments(mPostAct=mPostAct, act_fn=activation)
    varlen_args = VarlenArguments(mCuSeqlensM=mCuSeqlensM, mCuSeqlensK=mCuSeqlensK, mAIdx=mAIdx)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled_gemm = cute.compile(
        gemm,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )

    from triton.testing import do_bench

    flops = 2 * m * n * k * l
    # Calculate memory bandwidth
    bytes_A = m * k * l * (a_dtype.width // 8)  # A tensor: (m, k, l)
    bytes_B = n * k * l * (b_dtype.width // 8)  # B tensor: (n, k, l)
    bytes_D = m * n * l * (d_dtype.width // 8)  # D tensor: (m, n, l)
    bytes_C = m * n * l * (c_dtype.width // 8) if c_dtype is not None else 0  # C tensor: (m, n, l)
    total_bytes = bytes_A + bytes_B + bytes_D + bytes_C  # Read A, B, C; Write D

    repeats = iterations
    warmup = warmup_iterations

    import time

    if not (varlen_m or varlen_k) and not gather_A:
        time.sleep(0.5)
        if a_dtype.width == 8:
            assert l == 1
            scale_ab = torch.ones((1,), dtype=torch.float32, device="cuda")
            fn_cublas = lambda: torch._scaled_mm(
                a_torch[:, :, 0],
                b_torch[:, :, 0].mT,
                scale_a=scale_ab,
                scale_b=scale_ab,
                out_dtype=torch.bfloat16,
                use_fast_accum=fp8_fast_accum,
            )
        else:
            if c_torch is None:
                fn_cublas = lambda: torch.matmul(
                    a_torch.permute(2, 0, 1), b_torch.permute(2, 0, 1).mT
                )
            else:
                c_torch_convert = c_torch.to(a_torch.dtype)  # In case C is in FP32
                fn_cublas = lambda: torch.baddbmm(
                    c_torch_convert.permute(2, 0, 1),
                    a_torch.permute(2, 0, 1),
                    b_torch.permute(2, 0, 1).mT,
                )
        timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
        tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
        print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    time.sleep(0.5)

    def fn():
        compiled_gemm(mA, mB, mD, mC, epi_args, scheduler_args, varlen_args, current_stream)
        if tile_count_semaphore is not None and varlen_m:
            tile_count_semaphore.zero_()

    timing = do_bench(fn, warmup=warmup, rep=repeats)
    # Idk why but for some cases the 1st run is much slower
    time.sleep(0.5)
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    tflops = flops / (timing * 1e9)  # Convert to TFlops
    gbps = total_bytes / (timing * 1e6)  # Convert to GB/s (1e9 for ms->s, 1e9 for B->GB)
    print(f"Cute-DSL Average time: {timing:.3f} ms, TFLOPS: {tflops:.1f}, GB/s: {gbps:.0f}")
    fn()

    if not (varlen_m or varlen_k) and not gather_A:
        time.sleep(0.5)
        timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
        tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
        print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")


if __name__ == "__main__":
    args = parse_arguments()
    run(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.d_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.d_major,
        args.c_major,
        args.warmup_iterations,
        args.iterations,
    )
