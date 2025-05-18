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
    # Check current TensorFlow installation
    print("Checking current TensorFlow installation...")
    run_command("pip show tensorflow")
    
    # Uninstall current TensorFlow
    print("\nUninstalling current TensorFlow...")
    run_command("pip uninstall -y tensorflow tensorflow-intel")
    
    # Install TensorFlow with GPU support
    print("\nInstalling TensorFlow with GPU support...")
    run_command("pip install tensorflow==2.10.0")
    
    # Check NVIDIA drivers
    print("\nChecking NVIDIA drivers...")
    run_command("nvidia-smi")
    
    # Install CUDA Toolkit compatible with TensorFlow 2.10
    print("\nInstalling CUDA Toolkit 11.2 and cuDNN...")
    print("Please download and install CUDA Toolkit 11.2 from: https://developer.nvidia.com/cuda-11.2.0-download-archive")
    print("Please download and install cuDNN 8.1.0 for CUDA 11.2 from: https://developer.nvidia.com/cudnn")
    
    # Set environment variables
    print("\nSetting environment variables...")
    if os.name == 'nt':  # Windows
        print("Add the following to your system environment variables:")
        print("CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2")
        print("Add to PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
        print("Add to PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\libnvvp")
        print("Add to PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\extras\\CUPTI\\lib64")
    else:  # Linux/Mac
        print("Add the following to your ~/.bashrc or ~/.zshrc:")
        print("export CUDA_HOME=/usr/local/cuda-11.2")
        print("export PATH=$CUDA_HOME/bin:$PATH")
        print("export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")
    
    print("\nAfter installing CUDA and cuDNN, restart your computer and run check_gpu.py again.")

if __name__ == "__main__":
    main()
