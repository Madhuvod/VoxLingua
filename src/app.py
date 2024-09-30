import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from components.video_processor import VideoProcessor
from components.speech_recognition import transcribe_audio
from components.text_translator import process_translation
from components.text_to_speech import text_to_speech
from components.voice_cloning2 import process_audio

def main():
    # Step 1: Download YouTube video and extract audio
    processor = VideoProcessor()
    youtube_url = input("Enter the YouTube video URL: ")
    audio_path, video_path = processor.process_youtube(youtube_url)

    if not audio_path:
        print("Failed to process YouTube audio. Exiting.")
        return

    print(f"Audio downloaded to: {audio_path}")
    print(f"Video (without audio) downloaded to: {video_path}")

    # Step 2: Transcribe the audio
    transcription_file = transcribe_audio(os.path.basename(audio_path))
    if not transcription_file:
        print("Transcription failed. Exiting.")
        return

    print(f"Transcription saved to: {transcription_file}")

    # Step 3: Translate the transcription
    target_language = input("Enter the target language code (e.g., 'es' for Spanish): ")
    translation_file = process_translation(transcription_file, target_language)
    if not translation_file:
        print("Translation failed. Exiting.")
        return

    print(f"Translation saved to: {translation_file}")

    # Step 4: Convert translated text to speech
    tts_output = text_to_speech(translation_file, target_language)
    if not tts_output:
        print("Text-to-speech conversion failed. Exiting.")
        return

    print(f"Text-to-speech output saved to: {tts_output}")

    # Step 5: Clone voice
    output_dir = os.path.join(parent_dir, "temp", "voice_cloned_output")
    os.makedirs(output_dir, exist_ok=True)
    
    cloned_audio_path = process_audio(audio_path, tts_output, output_dir)
    if not cloned_audio_path:
        print("Voice cloning failed. Exiting.")
        return

    print(f"Voice cloned audio saved to: {cloned_audio_path}")
    print("Process completed successfully!")

if __name__ == "__main__":
    main()