import json
import os
import re

def read_transcriptions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    
    transcriptions = []
    for line in lines:
        if "|" in line:
            _, text = line.split("|", 1)
            transcriptions.append(text.strip())
    
    return transcriptions

def create_vocab(transcriptions):
    # Special tokens
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "|": 2,  # Delimiter token
    }
    
    # Add all unique characters from transcriptions
    next_id = len(vocab)
    for text in transcriptions:
        for char in text:
            if char not in vocab:
                vocab[char] = next_id
                next_id += 1
    
    return vocab

def main():
    transcriptions_file = "./dataset/transcriptions.txt"
    output_file = "./vocab.json"
    
    print(f"Reading transcriptions from {transcriptions_file}...")
    transcriptions = read_transcriptions(transcriptions_file)
    print(f"Found {len(transcriptions)} transcriptions.")
    
    print("Creating vocabulary...")
    vocab = create_vocab(transcriptions)
    print(f"Created vocabulary with {len(vocab)} tokens.")
    
    print(f"Saving vocabulary to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("Done!")
    
    # Print some statistics
    print("\nVocabulary statistics:")
    print(f"Total tokens: {len(vocab)}")
    print(f"Special tokens: {list(vocab.keys())[:3]}")
    print(f"First 10 character tokens: {list(vocab.keys())[3:13]}")

if __name__ == "__main__":
    main()
