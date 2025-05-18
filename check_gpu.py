import tensorflow as tf

# Check if TensorFlow can see the GPU
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Print GPU details if available
if len(tf.config.list_physical_devices('GPU')) > 0:
    # Get GPU device name
    gpu_device = tf.test.gpu_device_name()
    print("GPU device name:", gpu_device)
    
    # Test GPU with a simple operation
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:", c)
        print("Operation executed on GPU successfully!")
else:
    print("No GPU found. Please check your TensorFlow installation and GPU drivers.")
