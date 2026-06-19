from pathlib import Path
import torch

print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("Device: MPS")
else:
    print("Device: CPU")

root = Path(__file__).resolve().parents[1]
print("Project root:", root)
print("Data folder exists:", (root / "data").exists())
print("Src folder exists:", (root / "src").exists())