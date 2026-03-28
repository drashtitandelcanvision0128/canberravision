import os
import sys

# Force GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Add CUDA libraries to PATH if they exist
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    r"C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR",
]

for path in cuda_paths:
    if os.path.exists(path):
        os.environ['PATH'] = path + ';' + os.environ.get('PATH', '')
        print(f"Added CUDA path: {path}")

# Test CUDA availability
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        
        # Test GPU memory
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        print("✅ GPU test tensor created successfully")
        print("🎉 GPU is ready for use!")
        
    else:
        print("❌ CUDA not available")
        
except Exception as e:
    print(f"Error testing CUDA: {e}")

print("\n" + "="*50)
print("GPU Configuration Complete!")
print("If CUDA is not available, please install:")
print("1. NVIDIA drivers from nvidia.com")
print("2. CUDA Toolkit from developer.nvidia.com/cuda-downloads")
print("="*50)
