from datasets import load_dataset
import os
import soundfile as sf
import numpy as np
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

# Load the dataset
quranic_dataset = load_dataset("RetaSy/quranic_audio_dataset")

# Specify the folder where you want to save the files
output_folder = "./dataset"
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
val_folder = os.path.join(output_folder, "val")

# Create the directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Clear the transcriptions files if they exist
train_transcriptions_file = os.path.join(train_folder, "transcriptions.txt")
test_transcriptions_file = os.path.join(test_folder, "transcriptions.txt")
val_transcriptions_file = os.path.join(val_folder, "transcriptions.txt")

for file_path in [train_transcriptions_file, test_transcriptions_file, val_transcriptions_file]:
    if os.path.exists(file_path):
        open(file_path, 'w', encoding='utf-8').close()  # Clear the file

# Define split ratios
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Collect all examples from the dataset
all_examples = []
for split in quranic_dataset:
    for example in quranic_dataset[split]:
        all_examples.append(example)

# Shuffle the examples
random.shuffle(all_examples)

# Calculate split sizes
total_files = len(all_examples)
train_size = int(total_files * train_ratio)
test_size = int(total_files * test_ratio)
val_size = total_files - train_size - test_size

# Split the examples
train_examples = all_examples[:train_size]
test_examples = all_examples[train_size:train_size + test_size]
val_examples = all_examples[train_size + test_size:]

print(f"Total examples: {total_files}")
print(f"Train split: {len(train_examples)} examples")
print(f"Test split: {len(test_examples)} examples")
print(f"Validation split: {len(val_examples)} examples")

# Function to process and save examples
def process_examples(examples, folder, transcriptions_file):
    processed = 0
    for example in examples:
        try:
            audio_file_path = example['audio']['path']
            aya = example['Aya']
            audio_array = example['audio']['array']
            sampling_rate = example['audio']['sampling_rate']

            # Save the audio file
            audio_output_path = os.path.join(folder, os.path.basename(audio_file_path))
            # Make sure the audio file has a proper extension
            if not audio_output_path.endswith(('.wav', '.mp3', '.flac')):
                audio_output_path += '.wav'

            # Use soundfile to properly save the audio with the correct format
            sf.write(audio_output_path, audio_array, sampling_rate)

            # Create the transcription line
            transcription_line = f"{os.path.basename(audio_output_path)}|{aya}\n"

            # Save the transcription to a text file
            with open(transcriptions_file, "a", encoding="utf-8") as f:
                f.write(transcription_line)

            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{len(examples)} files in {os.path.basename(folder)}")

        except Exception as e:
            print(f"Error processing file in {os.path.basename(folder)}: {e}")

    return processed

# Process each split
try:
    print("\nProcessing train split...")
    train_processed = process_examples(train_examples, train_folder, train_transcriptions_file)

    print("\nProcessing test split...")
    test_processed = process_examples(test_examples, test_folder, test_transcriptions_file)

    print("\nProcessing validation split...")
    val_processed = process_examples(val_examples, val_folder, val_transcriptions_file)

    total_processed = train_processed + test_processed + val_processed
    print(f"\nCompleted! Successfully processed {total_processed}/{total_files} files.")
    print(f"Train: {train_processed}/{len(train_examples)}")
    print(f"Test: {test_processed}/{len(test_examples)}")
    print(f"Validation: {val_processed}/{len(val_examples)}")

except Exception as e:
    print(f"An error occurred: {e}")
