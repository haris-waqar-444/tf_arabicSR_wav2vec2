import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
    inputs = Input(shape=(AUDIO_MAXLEN,))
    hidden_states = Wav2Vec2Layer()(inputs)
    outputs = Dense(config.vocab_size)(hidden_states)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Load weights
    try:
        model.load_weights("finetuned-wav2vec2.keras")
        return model, None
    except Exception as e:
        return None, str(e)

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
            raise ValueError(f"Could not load audio file: {e}")
    
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
    try:
        # Preprocess audio
        processed_audio = preprocess_audio(audio_file)
        
        # Run inference
        logits = model(processed_audio, training=False)
        predictions = tf.argmax(logits, axis=-1)
        
        # Decode predictions
        pred = predictions.numpy()[0]
        # Filter out padding
        pred = [p for p in pred if p > 0]
        
        # Convert to text
        transcription = tokenizer.decode(pred)
        
        return transcription, None
    except Exception as e:
        return None, str(e)

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Speech Recognition")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        self.model = None
        self.audio_file = None
        
        self.create_widgets()
        self.load_model_async()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Arabic Speech Recognition", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Loading model...", font=("Arial", 10))
        self.status_label.pack(fill=tk.X)
        
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=5)
        self.progress.start()
        
        # File selection frame
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_label = ttk.Label(file_frame, text="No file selected", font=("Arial", 10))
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side=tk.RIGHT, padx=5)
        self.browse_button.state(["disabled"])
        
        # Transcription frame
        transcription_frame = ttk.LabelFrame(main_frame, text="Transcription", padding="10")
        transcription_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.transcription_text = tk.Text(transcription_frame, wrap=tk.WORD, font=("Arial", 12))
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.transcribe_button = ttk.Button(button_frame, text="Transcribe", command=self.transcribe)
        self.transcribe_button.pack(side=tk.RIGHT, padx=5)
        self.transcribe_button.state(["disabled"])
        
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT, padx=5)
    
    def load_model_async(self):
        def load():
            model, error = load_model()
            
            if model:
                self.model = model
                self.root.after(0, lambda: self.update_status("Model loaded successfully", True))
            else:
                self.root.after(0, lambda: self.update_status(f"Error loading model: {error}", False))
        
        import threading
        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()
    
    def update_status(self, message, success=True):
        self.status_label.config(text=message)
        self.progress.stop()
        self.progress.pack_forget()
        
        if success:
            self.browse_button.state(["!disabled"])
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")]
        )
        
        if file_path:
            self.audio_file = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.transcribe_button.state(["!disabled"])
    
    def transcribe(self):
        if not self.audio_file:
            messagebox.showerror("Error", "Please select an audio file first")
            return
        
        # Disable buttons during transcription
        self.transcribe_button.state(["disabled"])
        self.browse_button.state(["disabled"])
        self.status_label.config(text="Transcribing...")
        self.progress.pack(fill=tk.X, pady=5)
        self.progress.start()
        
        def process():
            transcription, error = transcribe_audio(self.model, self.audio_file)
            
            if transcription:
                self.root.after(0, lambda: self.show_transcription(transcription))
            else:
                self.root.after(0, lambda: self.show_error(error))
        
        import threading
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def show_transcription(self, transcription):
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, transcription)
        
        self.status_label.config(text="Transcription complete")
        self.progress.stop()
        self.progress.pack_forget()
        
        # Re-enable buttons
        self.transcribe_button.state(["!disabled"])
        self.browse_button.state(["!disabled"])
    
    def show_error(self, error):
        messagebox.showerror("Error", f"Transcription failed: {error}")
        
        self.status_label.config(text="Transcription failed")
        self.progress.stop()
        self.progress.pack_forget()
        
        # Re-enable buttons
        self.transcribe_button.state(["!disabled"])
        self.browse_button.state(["!disabled"])
    
    def clear(self):
        self.transcription_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
