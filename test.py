import os
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import random
import numpy as np
from arabic_processor import ArabicWav2Vec2Processor
from tensorflow.keras.callbacks import Callback, EarlyStopping # type: ignore
from tensorflow.keras.layers import Layer # type: ignore
from arabic_config import ArabicWav2Vec2Config
from arabic_losses import ArabicCTCLoss

# Check for GPU availability
print("Checking for GPU availability...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s):")
    for device in physical_devices:
        print(f"  - {device}")

    # Configure TensorFlow to use the first GPU
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Memory growth enabled for {physical_devices[0]}")

        # Set memory limit for 4GB GPU (RTX 1650)
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]  # Limit to 3GB (leaving 1GB for system)
        )
        print("GPU memory limit set to 3GB")
    except Exception as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Training will run on CPU.")

# Set mixed precision policy for faster training on compatible GPUs
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy set to: {policy.name}")
    print(f"Compute dtype: {policy.compute_dtype}")
    print(f"Variable dtype: {policy.variable_dtype}")
except Exception as e:
    print(f"Mixed precision policy setting failed: {e}")
    print("Using default precision.")

config = ArabicWav2Vec2Config()  # Using our custom Arabic config

# Constants
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2  # Reduced batch size for 4GB GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients to simulate larger batch size

# Set TensorFlow to use the GPU more efficiently
tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
print("XLA optimization enabled")

# Custom callback for gradient accumulation
class GradientAccumulationCallback(Callback):
    def __init__(self, accumulation_steps=1):
        super().__init__()
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulation = []
        self.batch_count = 0

    def on_train_begin(self, logs=None):
        # Get the trainable weights
        self.trainable_weights = self.model.trainable_weights
        # Initialize gradient accumulation
        self.gradient_accumulation = [tf.zeros_like(w) for w in self.trainable_weights]

    def on_train_batch_end(self, batch, logs=None):
        # Get the gradients
        self.batch_count += 1
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.trainable_weights)

        # Accumulate gradients
        self.gradient_accumulation = [acc + grad for acc, grad in zip(self.gradient_accumulation, gradients)]

        # Apply accumulated gradients after specified number of steps
        if self.batch_count == self.accumulation_steps:
            # Scale the gradients
            scaled_gradients = [grad / self.accumulation_steps for grad in self.gradient_accumulation]

            # Apply gradients
            self.model.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_weights))

            # Reset accumulation
            self.gradient_accumulation = [tf.zeros_like(w) for w in self.trainable_weights]
            self.batch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        # Apply any remaining gradients at the end of the epoch
        if self.batch_count > 0:
            # Scale the gradients
            scaled_gradients = [grad / self.batch_count for grad in self.gradient_accumulation]

            # Apply gradients
            self.model.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_weights))

            # Reset accumulation
            self.gradient_accumulation = [tf.zeros_like(w) for w in self.trainable_weights]
            self.batch_count = 0

@tf.keras.saving.register_keras_serializable(package="CustomLayers")
class Wav2Vec2Layer(Layer):
    def __init__(self, **kwargs):
        super(Wav2Vec2Layer, self).__init__(**kwargs)
        self.pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True, input_shape=(AUDIO_MAXLEN,))

    def call(self, inputs):
        return self.pretrained_layer(inputs)

    def get_config(self):
        config = super(Wav2Vec2Layer, self).get_config()
        return config

# Define your model
inputs = tf.keras.Input(shape=(AUDIO_MAXLEN,))
hidden_states = Wav2Vec2Layer()(inputs)
outputs = tf.keras.layers.Dense(config.vocab_size)(hidden_states)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model(tf.random.uniform(shape=(BATCH_SIZE, AUDIO_MAXLEN)))

print(model.summary())

LEARNING_RATE = 5e-5

loss_fn = ArabicCTCLoss(config, (BATCH_SIZE, AUDIO_MAXLEN), division_factor=BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# Define data directories for train and test
train_dir = "dataset/train"
test_dir = "dataset/test"

# Function to read transcription file
def read_txt_file(f):
    with open(f, "r", encoding="utf-8") as f:
        samples = f.read().strip().split("\n")
        samples_dict = {}
        for s in samples:
            if s and "|" in s:
                file_path, transcription = s.split("|", 1)
                # Extract file ID from the path (filename without extension)
                file_id = os.path.basename(file_path)[:-4]  # Remove .wav extension
                samples_dict[file_id] = transcription.strip()
    return samples_dict

REQUIRED_SAMPLE_RATE = 16000

def read_wav_file(file_path):
    try:
        # Pass the file path directly to sf.read instead of opening the file first
        audio, sample_rate = sf.read(file_path)
        if sample_rate != REQUIRED_SAMPLE_RATE:
            raise ValueError(
                f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
            )
        file_id = os.path.split(file_path)[-1][:-len(".wav")]
        return {file_id: audio}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

# Function to fetch sound-text mapping from a directory
def fetch_sound_text_mapping(directory):
    # Get all files in the directory
    all_files = os.listdir(directory)

    # Get wav files
    wav_files = [os.path.join(directory, f) for f in all_files if f.endswith(".wav")]

    # Get transcription file
    transcription_file = os.path.join(directory, "transcriptions.txt")

    # Read transcriptions
    txt_samples = read_txt_file(transcription_file)

    # Read audio files
    speech_samples = {}
    for f in wav_files:
        result = read_wav_file(f)
        if result:  # Only update if the file was read successfully
            speech_samples.update(result)

    # Create samples list
    samples = []
    for file_id in speech_samples.keys():
        if file_id in txt_samples and len(speech_samples[file_id]) < AUDIO_MAXLEN:
            samples.append((speech_samples[file_id], txt_samples[file_id]))

    return samples

# Fetch train and test samples
train_samples = fetch_sound_text_mapping(train_dir)
test_samples = fetch_sound_text_mapping(test_dir)

print(f"Loaded {len(train_samples)} training samples")
print(f"Loaded {len(test_samples)} testing samples")

# Display a sample
if train_samples:
    sample_speech, sample_text = train_samples[0]
    print(f"Sample text: {sample_text}")
    print(f"Sample audio shape: {sample_speech.shape}")


tokenizer = ArabicWav2Vec2Processor(is_tokenizer=True)
processor = ArabicWav2Vec2Processor(is_tokenizer=False)

def preprocess_text(text):
  label = tokenizer(text)
  # Ensure label is int32 for CTC loss
  return tf.cast(tf.constant(label), dtype=tf.int32)

def preprocess_speech(audio):
  # Convert audio to float32 for processing
  audio = tf.cast(tf.constant(audio), dtype=tf.float32)
  # Process and return the audio
  return processor(audio)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
tf.random.set_seed(SEED)

# Create a custom dataset with explicit types
def create_dataset(samples, batch_size, audio_maxlen, label_maxlen):
    def generator():
        for speech, text in samples:
            # Process speech and text
            processed_speech = preprocess_speech(speech)
            processed_text = preprocess_text(text)

            # Pad or truncate speech to fixed length
            if len(processed_speech) > audio_maxlen:
                processed_speech = processed_speech[:audio_maxlen]
            else:
                pad_length = audio_maxlen - len(processed_speech)
                processed_speech = tf.pad(processed_speech, [[0, pad_length]])

            # Ensure text is not too long
            if len(processed_text) > label_maxlen:
                processed_text = processed_text[:label_maxlen]

            yield processed_speech, processed_text

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(audio_maxlen,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    # Shuffle the dataset
    dataset = dataset.shuffle(len(samples))

    # Batch the dataset
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([audio_maxlen], [label_maxlen]),
        padding_values=(0.0, 0)
    )

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Create the train and test datasets
train_dataset = create_dataset(train_samples, BATCH_SIZE, AUDIO_MAXLEN, LABEL_MAXLEN)
test_dataset = create_dataset(test_samples, BATCH_SIZE, AUDIO_MAXLEN, LABEL_MAXLEN)

# Count batches
train_batches = sum(1 for _ in train_dataset)
test_batches = sum(1 for _ in test_dataset)

print(f"Train dataset has {train_batches} batches")
print(f"Test dataset has {test_batches} batches")

# Compile the model
model.compile(optimizer, loss=loss_fn)

# Define number of epochs
NUM_EPOCHS = 1000

# Create gradient accumulation callback
gradient_accumulation_callback = GradientAccumulationCallback(accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

# Create a callback to clear GPU memory after each epoch
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Clear GPU memory
        tf.keras.backend.clear_session()
        # Force garbage collection
        import gc
        gc.collect()
        print(f"Memory cleaned after epoch {epoch+1}")

# Create memory cleanup callback
memory_cleanup_callback = MemoryCleanupCallback()

# Create early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Stop training if no improvement after 2 epochs
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
    min_delta=0.001,  # Minimum change to qualify as an improvement
    verbose=1  # Print messages when early stopping is triggered
)

# Train the model
print(f"\nTraining the model for up to {NUM_EPOCHS} epochs...")
print(f"Using gradient accumulation with {GRADIENT_ACCUMULATION_STEPS} steps (effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"Early stopping will trigger if validation loss doesn't improve for {early_stopping_callback.patience} epochs")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=NUM_EPOCHS,
    steps_per_epoch=train_batches,  # Explicitly set the number of steps per epoch
    validation_steps=test_batches,  # Explicitly set the number of validation steps
    callbacks=[gradient_accumulation_callback, memory_cleanup_callback, early_stopping_callback],
    verbose=1
)

# Print training results
print("\nTraining results:")
actual_epochs = len(history.history['loss'])
for epoch in range(actual_epochs):
    print(f"Epoch {epoch+1}/{actual_epochs}:")
    print(f"  Train Loss: {history.history['loss'][epoch]:.4f}")
    print(f"  Val Loss: {history.history['val_loss'][epoch]:.4f}")

# Print early stopping information if training was stopped early
if actual_epochs < NUM_EPOCHS:
    print(f"\nEarly stopping was triggered after {actual_epochs} epochs")
    print(f"Best model weights from epoch {np.argmin(history.history['val_loss']) + 1} have been restored")

# Save the model
save_dir = "finetuned-wav2vec2.keras"
model.save(save_dir, include_optimizer=False)
print(f"\nModel saved to {save_dir}")