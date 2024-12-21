import os
import sys
import json
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer

# Path to your Vosk model directory
MODEL_PATH = "vosk-model-small-en-us-0.15"  # Update with your model's folder path

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Please download it from https://alphacephei.com/vosk/models")
    exit(1)

# Load the Vosk model
model = Model(MODEL_PATH)

# Initialize the recognizer with the model and sample rate
recognizer = KaldiRecognizer(model, 16000)

# String to save the transcribed speech
transcribed_text = ""
stop_flag = False  # Flag to stop the program after one full recognition

# Callback function to process audio in real-time
def callback(indata, frames, time, status):
    global transcribed_text, stop_flag
    
    if status:
        print(f"Audio error: {status}", file=sys.stderr)

    # Convert the numpy array to bytes (necessary for Vosk)
    audio_data = indata.astype(np.int16).tobytes()

    # Try to recognize speech
    if recognizer.AcceptWaveform(audio_data):
        result = json.loads(recognizer.Result())
        transcribed_text = result.get('text', '')  # Save full text in transcribed_text
        print(f"Recognized text: {transcribed_text}")
        stop_flag = True  # Set flag to True to stop the program
    else:
        partial_result = json.loads(recognizer.PartialResult())
        print(f"Partial text: {partial_result.get('partial', '')}")

# Configure audio stream settings
try:
    print("Listening... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=2048, callback=callback):
        while not stop_flag:  # Keep the stream open until a full result is recognized
            sd.sleep(100)  # Sleep for a short duration to allow the callback to run
except KeyboardInterrupt:
    print("\nExiting transcription...")

    # When you exit, you can print or save the transcribed string
    print(f"Final Transcribed Text: {transcribed_text}")
except Exception as e:
    print(f"An error occurred: {e}")