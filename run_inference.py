import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import numpy as np
import librosa
from arabic_processor import ArabicWav2Vec2Processor
from arabic_config import ArabicWav2Vec2Config

# Import Keras layers
Layer = tf.keras.layers.Layer
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense

# Define the Wav2Vec2Layer class (same as in training)
class Wav2Vec2Layer(Layer):
    def __init__(self, **kwargs):
        super(Wav2Vec2Layer, self).__init__(**kwargs)
        self.pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=False, input_shape=(246000,))

    def call(self, inputs):
        return self.pretrained_layer(inputs)

# Constants
AUDIO_MAXLEN = 246000
REQUIRED_SAMPLE_RATE = 16000

# Load configuration
config = ArabicWav2Vec2Config()

# Initialize processors
tokenizer = ArabicWav2Vec2Processor(is_tokenizer=True)
processor = ArabicWav2Vec2Processor(is_tokenizer=False)

def load_model():
    """Load the trained model"""
    print("Creating model architecture...")
    inputs = Input(shape=(AUDIO_MAXLEN,))
    hidden_states = Wav2Vec2Layer()(inputs)
    outputs = Dense(config.vocab_size)(hidden_states)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Load weights
    try:
        model.load_weights("finetuned-wav2vec2.keras")
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
    
    return model

def preprocess_audio(file_path, target_length=AUDIO_MAXLEN):
    """Preprocess audio file for inference"""
    print(f"Loading audio file: {file_path}")
    
    # Load audio file
    try:
        # Try loading with soundfile first
        audio, sample_rate = sf.read(file_path)
    except Exception as sf_error:
        print(f"Soundfile error: {sf_error}, trying librosa...")
        # If soundfile fails, try librosa
        try:
            audio, sample_rate = librosa.load(file_path, sr=None)
        except Exception as librosa_error:
            print(f"Could not load audio file: {librosa_error}")
            sys.exit(1)
    
    print(f"Original sample rate: {sample_rate}")
    
    # Resample if needed
    if sample_rate != REQUIRED_SAMPLE_RATE:
        print(f"Resampling from {sample_rate} to {REQUIRED_SAMPLE_RATE}")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=REQUIRED_SAMPLE_RATE)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print("Converting stereo to mono")
        audio = np.mean(audio, axis=1)
    
    # Convert to float32
    audio = tf.cast(tf.constant(audio), dtype=tf.float32)
    
    # Pad or truncate to target length
    current_length = audio.shape[0]
    print(f"Audio length: {current_length} samples ({current_length/REQUIRED_SAMPLE_RATE:.2f} seconds)")
    
    if current_length > target_length:
        print(f"Truncating audio to {target_length} samples")
        audio = audio[:target_length]
    elif current_length < target_length:
        print(f"Padding audio to {target_length} samples")
        padding = tf.zeros(target_length - current_length, dtype=tf.float32)
        audio = tf.concat([audio, padding], axis=0)
    
    # Process the audio
    processed_audio = processor(audio)
    
    # Add batch dimension
    processed_audio = tf.expand_dims(processed_audio, 0)
    
    return processed_audio

def transcribe_audio(model, audio_file):
    """Transcribe audio file using the model"""
    # Preprocess audio
    processed_audio = preprocess_audio(audio_file)
    
    # Run inference
    print("Running inference...")
    logits = model(processed_audio, training=False)
    predictions = tf.argmax(logits, axis=-1)
    
    # Decode predictions
    pred = predictions.numpy()[0]
    # Filter out padding
    pred = [p for p in pred if p > 0]
    
    # Convert to text
    transcription = tokenizer.decode(pred)
    
    return transcription

def main():
    # Path to your audio file
    audio_file = r"C:\Users\haris\OneDrive\Documents\Sound Recordings\Recording.wav"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        return
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Transcribe audio
    print("\nTranscribing audio...")
    transcription = transcribe_audio(model, audio_file)
    
    # Print results
    print("\n=== Transcription Results ===")
    print(f"Audio file: {audio_file}")
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()
