import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)


def test_neural_network():
    print("=" * 60)
    print("NEURAL NETWORK GPU TEST")
    print("=" * 60)

    # Check CUDA
    assert torch.cuda.is_available(), "CUDA is not available."

    device = torch.device("cuda:0")

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Device  : {device}")

    # Create neural network
    model = SimpleNetwork()

    # Move model to GPU
    model = model.to(device)

    print("\nModel device:")
    print(next(model.parameters()).device)

    # Create sample input
    batch_size = 8

    x = torch.randn(
        batch_size,
        128,
        device=device
    )

    # Fake labels
    # 0 = real
    # 1 = fake
    y = torch.randint(
        0,
        2,
        (batch_size,),
        device=device
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("\nStarting training test...")

    for epoch in range(5):

        optimizer.zero_grad()

        # Forward pass
        predictions = model(x)

        # Calculate loss
        loss = criterion(predictions, y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        print(
            f"Epoch {epoch + 1}/5 "
            f"- Loss: {loss.item():.4f}"
        )

    torch.cuda.synchronize()

    # Verify model is actually on GPU
    assert next(model.parameters()).is_cuda

    # Verify output shape
    assert predictions.shape == (batch_size, 2)

    print("\nGPU memory used:")
    print(
        f"{torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
    )

    print("\n" + "=" * 60)
    print("NEURAL NETWORK GPU TEST: PASSED")
    print("=" * 60)