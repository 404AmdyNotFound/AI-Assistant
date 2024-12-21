import io
import os
import sys
import json
from elevenlabs import ElevenLabs
from pydub import AudioSegment 
import simpleaudio as simpaud
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import threading
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer


MODEL_PATH = "vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

# Initialize the recognizer with the model and sample rate
recognizer = KaldiRecognizer(model, 16000)

# String to save the transcribed speech
transcribed_text = ""
stop_flag = False  # Flag to stop the program after one full recognition

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
    return transcribed_text

ElevenLabsAPIKEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ElevenLabsAPIKEY)
print(ElevenLabsAPIKEY)
CONTEXT_FILE = "convo_history.txt"
model = OllamaLLM(model="llama3")
template = """
Answer the question below.

Here is the conversation history: 
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def load_context():
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r") as file:
            return file.read()
    return ""

def save_context(context):
    with open(CONTEXT_FILE, "w") as file:
        file.write(context)

def talk(result):
    print(result)
    audio_generator = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",  # Your voice ID
    model_id="eleven_multilingual_v2",  # Use the multilingual model
    text=result  # The text you want to convert to speech
    )

    audio = b''.join(audio_generator)
    audio_segment = AudioSegment.from_mp3(io.BytesIO(audio))
    # Convert to raw audio data (PCM format)
    raw_audio = np.array(audio_segment.get_array_of_samples())

    # Ensure correct playback format
    if audio_segment.channels == 2:  # stereo
        raw_audio = raw_audio.reshape((-1, 2))  # stereo format
    else:  # mono
        raw_audio = raw_audio.reshape((-1, 1))  # mono format
    # Play the audio using sounddevice
    sd.play(raw_audio, samplerate=audio_segment.frame_rate)
    sd.wait()
    sd.sleep(400)


def handle_conversation(context):
    global transcribed_text, stop_flag
    while True:
            
            print("Listening... Press Ctrl+C to stop.")
            with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=2048, callback=callback):
                while not stop_flag:  # Keep the stream open until a full result is recognized
                    sd.sleep(100)  # Sleep for a short duration to allow the callback to run

            if (transcribed_text.lower() == "stop session"):
                print("Ending session...")
                save_context(context)
                break
            result = chain.invoke({"context": context, "question": transcribed_text})
            talk(result)
            context += f"\nUser: {transcribed_text}\nAI: {result}"
            stop_flag = False

if __name__ == "__main__":
    context = load_context()
    print("Welcome to the AI ChatBot! Type 'bye' to quit.")
    handle_conversation(context)
    
