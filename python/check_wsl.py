import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import sys
print(f"Python: {sys.executable}")

# Test if transformers works
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("transformers: OK")
except Exception as e:
    print(f"transformers: FAIL ({e})")
