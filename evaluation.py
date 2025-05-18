import os
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import random
from arabic_processor import ArabicWav2Vec2Processor
from arabic_config import ArabicWav2Vec2Config
from jiwer import wer

# Import Keras layers
Layer = tf.keras.layers.Layer
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense

# Define the Wav2Vec2Layer class (same as in training)
class Wav2Vec2Layer(Layer):
    def __init__(self, **kwargs):
        super(Wav2Vec2Layer, self).__init__(**kwargs)
        self.pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True, input_shape=(246000,))

    def call(self, inputs):
        return self.pretrained_layer(inputs)

# Constants
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2

# Load configuration
config = ArabicWav2Vec2Config()

# Initialize processors
tokenizer = ArabicWav2Vec2Processor(is_tokenizer=True)
processor = ArabicWav2Vec2Processor(is_tokenizer=False)

# Function to preprocess audio
def preprocess_audio(file_path, target_length=AUDIO_MAXLEN):
    audio, sample_rate = sf.read(file_path)
    if sample_rate != 16000:
        raise ValueError(f"Sample rate must be 16000, got {sample_rate}")

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

    return processed_audio

# Function to preprocess text
def preprocess_text(text):
    label = tokenizer(text)
    return tf.cast(tf.constant(label), dtype=tf.int32)

# This function has been replaced by create_dataset_from_samples

# Load the model
def load_model():
    # Create the model architecture
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

    return model

# Load the validation dataset
def load_validation_dataset():
    val_dir = "dataset/val"
    all_files = os.listdir(val_dir)

    wav_files = [f for f in all_files if f.endswith(".wav")]

    # Read transcriptions
    transcription_file = os.path.join(val_dir, "transcriptions.txt")
    transcriptions = {}
    with open(transcription_file, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                file_path, text = line.strip().split("|", 1)
                file_id = os.path.basename(file_path)
                transcriptions[file_id] = text

    # Create samples
    samples = []
    for wav_file in wav_files:
        file_path = os.path.join(val_dir, wav_file)
        if wav_file in transcriptions:
            samples.append((file_path, transcriptions[wav_file]))

    print(f"Loaded {len(samples)} validation samples")
    return samples

# Evaluation function
def evaluate_model(model, dataset, num_batches):
    predictions_list = []
    references_list = []

    for speech, labels in dataset.take(num_batches):
        logits = model(speech, training=False)
        predictions = tf.argmax(logits, axis=-1)

        # Convert to text
        for pred, label in zip(predictions.numpy(), labels.numpy()):
            # Filter out padding
            pred = [p for p in pred if p > 0]
            label = [l for l in label if l > 0]

            pred_text = tokenizer.decode(pred)
            label_text = tokenizer.decode(label, group_tokens=False)

            predictions_list.append(pred_text)
            references_list.append(label_text)

    # Calculate WER
    error_rate = wer(references_list, predictions_list)

    return {
        "wer": error_rate,
        "samples": list(zip(predictions_list, references_list))
    }

# Create a dataset from samples
def create_dataset_from_samples(samples, batch_size, audio_maxlen, label_maxlen):
    def generator():
        for speech, text in samples:
            # Process speech and text
            processed_speech = preprocess_audio(speech)
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

    # Batch the dataset
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([audio_maxlen], [label_maxlen]),
        padding_values=(0.0, 0)
    )

    return dataset

# Main evaluation
if __name__ == "__main__":
    print("Loading model...")
    model = load_model()

    print("Loading validation dataset...")
    val_samples = load_validation_dataset()

    print("Creating validation dataset...")
    val_dataset = create_dataset_from_samples(val_samples, BATCH_SIZE, AUDIO_MAXLEN, LABEL_MAXLEN)

    # Count batches
    num_val_batches = sum(1 for _ in val_dataset)
    print(f"Validation dataset has {num_val_batches} batches")

    print("Evaluating model...")
    results = evaluate_model(model, val_dataset, num_val_batches)

    print(f"Word Error Rate (WER): {results['wer']:.4f}")

    print("\nSample predictions:")
    for i, (pred, ref) in enumerate(results['samples'][:5]):
        print(f"Example {i+1}:")
        print(f"  Reference: {ref}")
        print(f"  Prediction: {pred}")
        print()
