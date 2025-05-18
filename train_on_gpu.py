import subprocess
import sys
import os
import time
from monitor_gpu import monitor_in_background

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.communicate()
    return process.returncode

def main():
    print("=== Training Wav2Vec2 Model on GPU ===")
    
    # Check if CUDA is available
    print("\nChecking GPU availability...")
    gpu_check = subprocess.run(['python', 'check_gpu.py'], capture_output=True, text=True)
    print(gpu_check.stdout)
    
    if "No GPU found" in gpu_check.stdout:
        print("\nNo GPU detected. Would you like to:")
        print("1. Install GPU support")
        print("2. Continue with CPU training")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            print("\nRunning GPU support installation script...")
            run_command("python install_gpu_support.py")
            print("\nPlease restart your computer and run this script again.")
            return
        elif choice == "2":
            print("\nContinuing with CPU training...")
        elif choice == "3":
            print("\nExiting...")
            return
        else:
            print("\nInvalid choice. Exiting...")
            return
    
    # Start GPU monitoring in background
    print("\nStarting GPU monitoring...")
    monitor_thread = monitor_in_background(interval=10, log_file="gpu_training_log.csv")
    
    # Run the training script
    print("\nStarting training...")
    start_time = time.time()
    
    run_command("python test.py")
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"GPU monitoring log saved to gpu_training_log.csv")
    
    # Run evaluation
    print("\nRunning evaluation...")
    run_command("python evaluation.py")
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()
