import os
import sys
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from arabic_processor import ArabicWav2Vec2Processor
from arabic_config import ArabicWav2Vec2Config
from audio_analyzer import load_audio, convert_to_mono, resample_audio, normalize_audio, denoise_audio, segment_audio, save_processed_audio

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

def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    inputs = Input(shape=(AUDIO_MAXLEN,))
    hidden_states = Wav2Vec2Layer()(inputs)
    outputs = Dense(config.vocab_size)(hidden_states)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Load weights
    try:
        model.load_weights(model_path)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
    
    return model

def preprocess_for_inference(audio, sr, target_length=AUDIO_MAXLEN):
    """Preprocess audio for inference"""
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

def transcribe_audio(model, audio, sr):
    """Transcribe audio using the model"""
    # Preprocess audio
    processed_audio = preprocess_for_inference(audio, sr)
    
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

def enhanced_transcription(file_path, model_path, output_dir=None, segment_audio_files=False):
    """Perform enhanced transcription with audio preprocessing"""
    print(f"Processing audio file: {file_path}")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load audio
    audio, sr = load_audio(file_path)
    print(f"Original sample rate: {sr} Hz")
    print(f"Audio duration: {len(audio) / sr:.2f} seconds")
    
    # Convert to mono if stereo
    audio = convert_to_mono(audio)
    
    # Resample to required sample rate
    if sr != REQUIRED_SAMPLE_RATE:
        audio = resample_audio(audio, sr, REQUIRED_SAMPLE_RATE)
        sr = REQUIRED_SAMPLE_RATE
    
    # Normalize audio
    audio_normalized = normalize_audio(audio)
    
    # Denoise audio
    audio_denoised = denoise_audio(audio_normalized, sr)
    
    # Save processed audio if output directory is provided
    if output_dir:
        processed_path = os.path.join(output_dir, "processed_audio.wav")
        save_processed_audio(audio_denoised, sr, processed_path)
    
    # Load model
    model = load_model(model_path)
    
    # Transcribe full audio
    print("\nTranscribing full audio...")
    full_transcription = transcribe_audio(model, audio_denoised, sr)
    
    results = {
        "full_audio": {
            "audio": audio_denoised,
            "transcription": full_transcription
        },
        "segments": []
    }
    
    # Segment audio and transcribe each segment
    print("\nSegmenting audio and transcribing each segment...")
    segments, boundaries = segment_audio(audio_denoised, sr)
    
    if segments:
        print(f"Found {len(segments)} speech segments")
        
        for i, segment in enumerate(segments):
            print(f"Transcribing segment {i+1}/{len(segments)}...")
            
            # Save segment if requested
            if output_dir and segment_audio_files:
                segment_path = os.path.join(output_dir, f"segment_{i+1}.wav")
                save_processed_audio(segment, sr, segment_path)
            
            # Transcribe segment
            segment_transcription = transcribe_audio(model, segment, sr)
            
            # Add to results
            start_time = boundaries[i][0] / sr
            end_time = boundaries[i][1] / sr
            
            results["segments"].append({
                "index": i+1,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "transcription": segment_transcription
            })
    
    # Print results
    print("\n=== Transcription Results ===")
    print(f"Audio file: {file_path}")
    print(f"Full transcription: {full_transcription}")
    
    if segments:
        print("\nSegment transcriptions:")
        for segment in results["segments"]:
            print(f"Segment {segment['index']} ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s): {segment['transcription']}")
    
    # Save results to file if output directory is provided
    if output_dir:
        results_path = os.path.join(output_dir, "transcription_results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"Audio file: {file_path}\n")
            f.write(f"Full transcription: {full_transcription}\n\n")
            
            if segments:
                f.write("Segment transcriptions:\n")
                for segment in results["segments"]:
                    f.write(f"Segment {segment['index']} ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s): {segment['transcription']}\n")
        
        print(f"\nResults saved to {results_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Arabic speech transcription")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--model", default="finetuned-wav2vec2.keras", help="Path to the model file")
    parser.add_argument("--output", help="Directory to save processed audio and transcription results")
    parser.add_argument("--save-segments", action="store_true", help="Save individual audio segments")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at {args.audio}")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    # Perform enhanced transcription
    enhanced_transcription(args.audio, args.model, args.output, args.save_segments)

if __name__ == "__main__":
    main()
