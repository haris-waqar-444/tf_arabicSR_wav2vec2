import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if stderr:
        print(f"Error: {stderr.decode()}")
    return process.returncode

def main():
    print("=== Installing TensorFlow GPU Support for RTX 1650 4GB ===")
    
    # Check current TensorFlow installation
    print("\nChecking current TensorFlow installation...")
    run_command("pip show tensorflow")
    
    # Uninstall current TensorFlow
    print("\nUninstalling current TensorFlow...")
    run_command("pip uninstall -y tensorflow tensorflow-intel tensorflow-io tensorflow-io-gcs-filesystem")
    
    # Install compatible TensorFlow version
    print("\nInstalling TensorFlow 2.10.0 (compatible with CUDA 11.2)...")
    run_command("pip install tensorflow==2.10.0")
    
    # Install CUDA Toolkit and cuDNN
    print("\n=== CUDA Toolkit and cuDNN Installation Instructions ===")
    print("1. Download and install CUDA Toolkit 11.2 from:")
    print("   https://developer.nvidia.com/cuda-11.2.0-download-archive")
    print("\n2. Download cuDNN 8.1.0 for CUDA 11.2 from:")
    print("   https://developer.nvidia.com/cudnn")
    print("   (You'll need to create an NVIDIA account)")
    print("\n3. Extract the cuDNN files and copy them to your CUDA installation:")
    print("   - Copy bin/* to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
    print("   - Copy include/* to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include")
    print("   - Copy lib/x64/* to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\lib\\x64")
    
    # Set environment variables
    print("\n=== Environment Variables Setup ===")
    print("Add the following to your system environment variables:")
    print("1. Add to PATH:")
    print("   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
    print("   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\libnvvp")
    print("   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\extras\\CUPTI\\lib64")
    print("\n2. Create new variable CUDA_PATH:")
    print("   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2")
    
    # Install additional packages
    print("\nInstalling additional packages...")
    run_command("pip install nvidia-cudnn-cu11==8.6.0.163")
    
    # Verify installation
    print("\n=== Verification ===")
    print("After installation, restart your computer and run:")
    print("python check_gpu.py")
    
    print("\nInstallation instructions completed!")

if __name__ == "__main__":
    main()
