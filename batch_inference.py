import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import numpy as np
import librosa
import argparse
import time
from tqdm import tqdm
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

def load_model(model_path="finetuned-wav2vec2.keras"):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    
    inputs = Input(shape=(AUDIO_MAXLEN,))
    hidden_states = Wav2Vec2Layer()(inputs)
    outputs = Dense(config.vocab_size)(hidden_states)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Load weights
    try:
        model.load_weights(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def preprocess_audio(file_path, target_length=AUDIO_MAXLEN):
    """Preprocess audio file for inference"""
    try:
        # Try loading with soundfile first
        audio, sample_rate = sf.read(file_path)
    except Exception:
        # If soundfile fails, try librosa
        try:
            audio, sample_rate = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
    
    # Resample if needed
    if sample_rate != REQUIRED_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=REQUIRED_SAMPLE_RATE)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Convert to float32
    audio = tf.cast(tf.constant(audio), dtype=tf.float32)
    
    # Pad or truncate to target length
    current_length = audio.shape[0]
    
    if current_length > target_length:
        audio = audio[:target_length]
    elif current_length < target_length:
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
    
    if processed_audio is None:
        return None
    
    # Run inference
    logits = model(processed_audio, training=False)
    predictions = tf.argmax(logits, axis=-1)
    
    # Decode predictions
    pred = predictions.numpy()[0]
    # Filter out padding
    pred = [p for p in pred if p > 0]
    
    # Convert to text
    transcription = tokenizer.decode(pred)
    
    return transcription

def process_directory(model, input_dir, output_file, extensions=None):
    """Process all audio files in a directory"""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    # Get all audio files
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    results = []
    start_time = time.time()
    
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        transcription = transcribe_audio(model, audio_file)
        if transcription is not None:
            results.append((audio_file, transcription))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("File,Transcription\n")
        for file_path, transcription in results:
            f.write(f"{file_path},{transcription}\n")
    
    print(f"Processed {len(results)} files in {processing_time:.2f} seconds")
    print(f"Results saved to {output_file}")

def process_single_file(model, input_file):
    """Process a single audio file"""
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    print(f"Processing file: {input_file}")
    start_time = time.time()
    
    transcription = transcribe_audio(model, input_file)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if transcription is not None:
        print(f"\nTranscription: {transcription}")
        print(f"Processing time: {processing_time:.2f} seconds")
    else:
        print("Transcription failed")

def main():
    parser = argparse.ArgumentParser(description="Batch inference for Arabic speech recognition")
    parser.add_argument("--input", required=True, help="Input audio file or directory")
    parser.add_argument("--output", help="Output CSV file for batch processing")
    parser.add_argument("--model", default="finetuned-wav2vec2.keras", help="Path to model file")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Process input
    if os.path.isdir(args.input):
        if args.output is None:
            args.output = "transcriptions.csv"
        process_directory(model, args.input, args.output)
    else:
        process_single_file(model, args.input)

if __name__ == "__main__":
    main()
