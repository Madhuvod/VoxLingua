import os
import sys
import gradio as gr

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from components.video_processor import VideoProcessor
from components.speech_recognition import transcribe_audio
from components.text_translator import process_translation
from components.text_to_speech import text_to_speech

def process_youtube_link(youtube_link, target_language, voice_pitch=1.0, voice_speed=1.0):
    # Step 1: Download YouTube video and extract audio
    processor = VideoProcessor()
    audio_path, video_path = processor.process_youtube(youtube_link)

    if not audio_path:
        return "Failed to process YouTube audio."

    # Step 2: Transcribe the audio
    transcription_file = transcribe_audio(os.path.basename(audio_path))
    if not transcription_file:
        return "Transcription failed."

    # Step 3: Translate the transcription
    translation_file = process_translation(transcription_file, target_language)
    if not translation_file:
        return "Translation failed."

    # Step 4: Convert translated text to speech with customized voice settings
    tts_output = text_to_speech(translation_file, target_language, pitch=voice_pitch, speed=voice_speed)
    if not tts_output:
        return "Text-to-speech conversion failed."
    
    return tts_output

# Define the Gradio interface
interface = gr.Interface(
    fn=process_youtube_link,
    inputs=[
        gr.Textbox(label="YouTube Link"),
        gr.Dropdown(
            choices=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh-CN", "ko"],  # Added more language options
            label="Target Language"
        ),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Voice Pitch (0.5-2.0)"),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Voice Speed (0.5-2.0)")
    ],
    outputs=gr.Audio(label="Translated Audio"),
    title="VoxLingua - YouTube Voice Translation",
    description="Upload a YouTube link, select a target language and customize voice settings to get the translated audio with preserved voice characteristics."
)

# Launch the interface with a public link
if __name__ == "__main__":
    interface.launch(share=True)