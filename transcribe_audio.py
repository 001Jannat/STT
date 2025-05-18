from transformers import pipeline

# Initialize the Whisper pipeline
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Path to your audio file
audio_path = r"C:\Users\VICTUS\OneDrive\Desktop\inter\love.mp3"

# Transcribe the audio with timestamps enabled
result = whisper_pipeline(audio_path, return_timestamps=True)

print("\n Transcribed Text:\n")

# If result is a list of chunks (with timestamps)
if isinstance(result, list):
    for chunk in result:
        text = chunk.get("text", "")
        start, end = chunk.get("timestamp", ["?", "?"])
        print(f"[{start} - {end}] {text}")
# If result is a single dict (no chunking)
elif isinstance(result, dict) and "text" in result:
    print(result["text"])
else:
    print("⚠️ No valid transcription found.")
