import subprocess
import time
import os
import sys
import threading
import datetime

def get_gpu_info():
    try:
        # Run nvidia-smi command
        result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu', '--format=csv,noheader'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        
        # Parse the output
        output = result.stdout.strip()
        return output
    except subprocess.CalledProcessError as e:
        return f"Error running nvidia-smi: {e}"
    except FileNotFoundError:
        return "nvidia-smi not found. Make sure NVIDIA drivers are installed."

def monitor_gpu(interval=5, log_file=None):
    """
    Monitor GPU usage at specified intervals
    
    Args:
        interval: Time between checks in seconds
        log_file: File to save the logs (optional)
    """
    print(f"Starting GPU monitoring (interval: {interval}s)")
    print("Press Ctrl+C to stop monitoring")
    
    if log_file:
        with open(log_file, 'w') as f:
            f.write("Timestamp,GPU Name,GPU Utilization (%),Memory Utilization (%),Total Memory (MB),Used Memory (MB),Free Memory (MB),Temperature (°C)\n")
    
    try:
        while True:
            # Get current time
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get GPU info
            gpu_info = get_gpu_info()
            
            # Print to console
            print(f"\n=== GPU Status at {current_time} ===")
            print(gpu_info)
            
            # Save to log file if specified
            if log_file and gpu_info:
                with open(log_file, 'a') as f:
                    f.write(f"{gpu_info}\n")
            
            # Wait for next check
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nGPU monitoring stopped")

def monitor_in_background(interval=5, log_file=None):
    """Start GPU monitoring in a background thread"""
    thread = threading.Thread(target=monitor_gpu, args=(interval, log_file), daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    # Default monitoring interval (seconds)
    interval = 5
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            print(f"Invalid interval: {sys.argv[1]}. Using default: {interval}s")
    
    # Default log file
    log_file = f"gpu_monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Start monitoring
    monitor_gpu(interval, log_file)
