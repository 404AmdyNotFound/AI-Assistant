import io
import os
import numpy as np
import sounddevice as sd
from elevenlabs import ElevenLabs
from pydub import AudioSegment

# Initialize ElevenLabs client
ElevenLabsAPIKEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ElevenLabsAPIKEY)

def text_to_speech_and_play(text):
    try:
        # Generate audio using ElevenLabs API
        audio_generator = client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_multilingual_v2",
            text=text
        )

        # Combine audio into a single stream
        audio_stream = b''.join(audio_generator)
        if not audio_stream:
            print("Error: Audio stream is empty!")
            return

        # Decode the MP3 audio stream to PCM using pydub
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_stream))
        
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

    except Exception as e:
        print(f"An error occurred: {e}")

# Continuous loop to get user input
while True:
    user_input = input("Say something (type 'bye' to exit): ")
    if user_input.lower() == "bye":
        print("Goodbye!")
        break
    text_to_speech_and_play(user_input)