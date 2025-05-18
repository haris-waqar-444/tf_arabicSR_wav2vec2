# Arabic Speech Recognition with Wav2Vec2

This repository contains code for fine-tuning a Wav2Vec2 model for Arabic speech recognition. The model is based on the pre-trained Wav2Vec2 model from TensorFlow Hub and is fine-tuned on Arabic speech data.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [TFLite Conversion](#tflite-conversion)
- [Troubleshooting](#troubleshooting)

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- TensorFlow Hub
- SoundFile
- Librosa
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/haris-waqar-444/tf_arabicSR_wav2vec2.git
   cd tf_arabicSR_wav2vec2
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `test.py`: Main script for training the model
- `inference.py`: Script for running inference on audio files
- `tflite.py`: Script for converting the model to TFLite format
- `arabic_processor.py`: Contains the ArabicWav2Vec2Processor class for processing audio and text
- `arabic_config.py`: Contains the configuration for the Arabic Wav2Vec2 model
- `arabic_losses.py`: Contains the CTC loss function for Arabic ASR
- `vocab.json`: Vocabulary file for the Arabic tokenizer

## Data Preparation

The model expects data in the following structure:

```
dataset/
├── train/
│   ├── audio1.wav
│   ├── audio2.wav
│   ├── ...
│   └── transcriptions.txt
└── test/
    ├── audio1.wav
    ├── audio2.wav
    ├── ...
    └── transcriptions.txt
```

The `transcriptions.txt` file should have the following format:
```
path/to/audio1.wav|transcription1
path/to/audio2.wav|transcription2
...
```

### Downloading Data

You can use any Arabic speech dataset. Here are some options:

1. [Common Voice Arabic](https://commonvoice.mozilla.org/en/datasets)
2. [Arabic Speech Corpus](https://www.openslr.org/93/)

After downloading, organize the data as described above.

## Training

To train the model, run:

```bash
python training.py
```

The script will:
1. Load and preprocess the training and testing data
2. Create a Wav2Vec2 model with a classification head
3. Train the model using CTC loss
4. Save the trained model to `finetuned-wav2vec2.keras`

### Training Parameters

You can modify the following parameters in `test.py`:
- `BATCH_SIZE`: Batch size for training (default: 2)
- `GRADIENT_ACCUMULATION_STEPS`: Number of steps to accumulate gradients (default: 4)
- `NUM_EPOCHS`: Maximum number of training epochs (default: 1000)
- `LEARNING_RATE`: Learning rate for the Adam optimizer (default: 5e-5)

The training includes:
- Gradient accumulation for simulating larger batch sizes
- Early stopping to prevent overfitting
- Memory cleanup after each epoch

## Evaluation

The model is evaluated on the test set during training. The validation loss is monitored for early stopping.

## Inference

To transcribe an audio file, run:

```bash
python inference.py
```

By default, the script uses a hardcoded path to an audio file. To change this, modify the `audio_file` variable in the `main()` function of `inference.py`.

The script will:
1. Load the trained model
2. Preprocess the audio file
3. Run inference to get the transcription
4. Print the transcription

### Custom Inference

You can also use the model in your own code:

```python
from inference import load_model, transcribe_audio

# Load the model
model = load_model()

# Transcribe an audio file
transcription = transcribe_audio(model, "path/to/audio.wav")
print(f"Transcription: {transcription}")
```

## TFLite Conversion

To convert the model to TFLite format for deployment on mobile or edge devices, run:

```bash
python tflite.py
```

This will:
1. Load the trained Keras model
2. Convert it to TFLite format
3. Save it as `ASR_wav2vec2.tflite`

## Troubleshooting

### GPU Memory Issues

If you encounter GPU memory issues, try:
- Reducing `BATCH_SIZE` in `test.py`
- Increasing `GRADIENT_ACCUMULATION_STEPS` to simulate larger batch sizes
- Setting a memory limit for your GPU in `test.py`

### Audio Format Issues

If you have issues with audio files, ensure:
- Audio files are in WAV format
- Sample rate is 16kHz (or will be resampled)
- Audio is mono (or will be converted)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

- [TensorFlow Hub](https://tfhub.dev/) for the pre-trained Wav2Vec2 model
- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477) for the original research