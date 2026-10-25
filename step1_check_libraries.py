"""Step 1: check that required libraries are installed. Run: python scripts/step1_check_libraries.py"""
import sys
import importlib

print(f"Python: {sys.version.split()[0]}")
for pkg, mod in [("PyTorch", "torch"), ("Ultralytics", "ultralytics"), ("OpenCV", "cv2"),
                 ("NumPy", "numpy"), ("Pillow", "PIL"), ("Streamlit", "streamlit")]:
    try:
        m = importlib.import_module(mod)
        print(f"[OK]      {pkg:12s} {getattr(m, '__version__', 'unknown')}")
    except ImportError:
        print(f"[MISSING] {pkg:12s} -> run: pip install -r requirements.txt")
try:
    import torch
    print("CUDA GPU available:", torch.cuda.is_available(), "(False is fine - this project runs on CPU)")
    print("Quick tensor test:", (torch.ones(2, 2) * 2).sum().item(), "(expected 8.0)")
except ImportError:
    pass
