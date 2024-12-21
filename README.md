# AI-Assistant

Implementing Ollama to create an AI assistant (no cost).
The goal of this project is to create a JARVIS like AI assistant which I can eventually run on a Microprocessor/Single-board Computer such as a Raspberry Pi.

API keys are required for: 
1. ElevenLabs

The used AI model is Llama 3. 
The ElevenLabs API is used for the voice of the chatbot (for TTS functionality).
The VOSK API was used for speech recognition (speech-to-text).

All libraries used:
1. elevenlabs - For text-to-speech functionality via the ElevenLabs API.
2. pydub - For processing audio files (MP3 to raw audio conversion).
3. simpleaudio - For playing raw audio (used in the chatbot example).
4. langchain_ollama - For interacting with the Ollama LLM.
5. langchain-core - For handling prompts and chaining them with the LLM.
6. numpy - For handling raw audio data and numerical operations.
7. sounddevice - For capturing and playing audio.
8. vosk - For speech recognition (transcription).
9. io - Part of the Python standard library (no installation needed).
10. os - Part of the Python standard library (no installation needed).
11. sys - Part of the Python standard library (no installation needed).