import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play

def load_audio(file_path):
    """Load audio file and return audio data and sample rate"""
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
    
    return audio, sample_rate

def convert_to_mono(audio):
    """Convert stereo audio to mono"""
    if len(audio.shape) > 1:
        print("Converting stereo to mono")
        audio = np.mean(audio, axis=1)
    return audio

def resample_audio(audio, orig_sr, target_sr):
    """Resample audio to target sample rate"""
    if orig_sr != target_sr:
        print(f"Resampling from {orig_sr} to {target_sr}")
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio

def normalize_audio(audio):
    """Normalize audio to range [-1, 1]"""
    return audio / np.max(np.abs(audio))

def denoise_audio(audio, sr):
    """Apply simple noise reduction"""
    # Get noise profile from the first 0.5 seconds
    noise_sample = audio[:int(sr * 0.5)]
    noise_profile = np.mean(np.abs(noise_sample))
    
    # Apply simple noise gate
    threshold = noise_profile * 2
    audio_denoised = audio.copy()
    audio_denoised[np.abs(audio) < threshold] = 0
    
    return audio_denoised

def segment_audio(audio, sr, min_silence_len=0.5, silence_thresh=-40):
    """Segment audio into speech parts by removing silence"""
    # Convert to pydub AudioSegment
    audio_float = audio * (2**15)  # Convert to 16-bit PCM range
    audio_int16 = audio_float.astype(np.int16)
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    
    # Split on silence
    chunks = librosa.effects.split(
        audio,
        top_db=abs(silence_thresh),
        frame_length=2048,
        hop_length=512
    )
    
    # Filter out chunks that are too short
    min_samples = int(min_silence_len * sr)
    chunks = [chunk for chunk in chunks if (chunk[1] - chunk[0]) >= min_samples]
    
    # Extract segments
    segments = []
    for start, end in chunks:
        segments.append(audio[start:end])
    
    return segments, chunks

def plot_audio(audio, sr, title="Audio Waveform"):
    """Plot audio waveform"""
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sr, title="Spectrogram"):
    """Plot audio spectrogram"""
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_processed_audio(audio, sr, output_path):
    """Save processed audio to file"""
    sf.write(output_path, audio, sr)
    print(f"Saved processed audio to {output_path}")

def analyze_audio(file_path, output_dir=None, plot=False):
    """Analyze audio file and return processed audio"""
    print(f"Analyzing audio file: {file_path}")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load audio
    audio, sr = load_audio(file_path)
    print(f"Original sample rate: {sr} Hz")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {librosa.get_duration(y=audio, sr=sr):.2f} seconds")
    
    # Convert to mono if stereo
    audio = convert_to_mono(audio)
    
    # Resample to 16kHz if needed (standard for speech recognition)
    target_sr = 16000
    audio = resample_audio(audio, sr, target_sr)
    sr = target_sr
    
    # Normalize audio
    audio_normalized = normalize_audio(audio)
    
    # Denoise audio
    audio_denoised = denoise_audio(audio_normalized, sr)
    
    # Segment audio
    segments, boundaries = segment_audio(audio_denoised, sr)
    print(f"Found {len(segments)} speech segments")
    
    # Plot if requested
    if plot:
        # Original waveform
        plot_audio(audio, sr, "Original Waveform")
        
        # Normalized waveform
        plot_audio(audio_normalized, sr, "Normalized Waveform")
        
        # Denoised waveform
        plot_audio(audio_denoised, sr, "Denoised Waveform")
        
        # Spectrogram
        plot_spectrogram(audio_denoised, sr, "Spectrogram")
        
        # Plot segments
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio_denoised, sr=sr)
        for start, end in boundaries:
            start_time = start / sr
            end_time = end / sr
            plt.axvline(x=start_time, color='r', linestyle='--')
            plt.axvline(x=end_time, color='r', linestyle='--')
        plt.title("Speech Segments")
        plt.tight_layout()
        plt.show()
    
    # Save processed audio if output directory is provided
    if output_dir:
        # Save full processed audio
        output_path = os.path.join(output_dir, "processed_audio.wav")
        save_processed_audio(audio_denoised, sr, output_path)
        
        # Save individual segments
        for i, segment in enumerate(segments):
            segment_path = os.path.join(output_dir, f"segment_{i+1}.wav")
            save_processed_audio(segment, sr, segment_path)
    
    return audio_denoised, sr, segments

def main():
    parser = argparse.ArgumentParser(description="Analyze and process audio files for speech recognition")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--output", help="Directory to save processed audio files")
    parser.add_argument("--plot", action="store_true", help="Plot audio waveforms and spectrograms")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at {args.audio}")
        return
    
    # Analyze audio
    analyze_audio(args.audio, args.output, args.plot)

if __name__ == "__main__":
    main()
