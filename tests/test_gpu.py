import torch


def test_gpu():
    """Verify that PyTorch can perform computation on the NVIDIA GPU."""

    print("=" * 60)
    print("GPU COMPUTATION TEST")
    print("=" * 60)

    # 1. Check CUDA availability
    assert torch.cuda.is_available(), (
        "CUDA is not available. Check the NVIDIA driver and "
        "PyTorch CUDA installation."
    )

    # 2. Select GPU
    device = torch.device("cuda:0")

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Device          : {device}")

    # 3. Create tensors on GPU
    print("\nCreating GPU tensors...")

    matrix_a = torch.randn(
        2048,
        2048,
        device=device
    )

    matrix_b = torch.randn(
        2048,
        2048,
        device=device
    )

    print(f"Matrix A device : {matrix_a.device}")
    print(f"Matrix B device : {matrix_b.device}")

    # 4. GPU matrix multiplication
    print("\nPerforming matrix multiplication...")

    matrix_c = torch.matmul(matrix_a, matrix_b)

    # Wait for GPU operation to finish
    torch.cuda.synchronize()

    print(f"Matrix C device : {matrix_c.device}")
    print(f"Matrix C shape  : {matrix_c.shape}")

    # 5. Check GPU memory
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    print(f"\nGPU memory allocated : {allocated:.2f} MB")
    print(f"GPU memory reserved  : {reserved:.2f} MB")

    # 6. Verify the result
    assert matrix_c.shape == (2048, 2048)
    assert matrix_c.device.type == "cuda"

    # 7. Clean up
    del matrix_a
    del matrix_b
    del matrix_c

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("GPU COMPUTATION TEST: PASSED")
    print("=" * 60)