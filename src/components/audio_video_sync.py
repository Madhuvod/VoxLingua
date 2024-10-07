import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from moviepy.video.fx.all import loop
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

def sync_audio_video(video_path, audio_dir, output_path):
    # Load the original video
    video_clip = VideoFileClip(video_path)
    
    # List all .mp3 audio files in the directory
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])
    
    # Load and concatenate all audio clips
    audio_clips = [AudioFileClip(os.path.join(audio_dir, audio_file)) for audio_file in audio_files]
    
    # Concatenate all audio clips into one final audio
    final_audio = concatenate_audioclips(audio_clips)
    
    # Check if the final audio is shorter or longer than the video
    if final_audio.duration < video_clip.duration:
        # Loop the audio to match video length
        final_audio = loop(final_audio, duration=video_clip.duration)
    elif final_audio.duration > video_clip.duration:
        # Trim the audio to fit the video duration
        final_audio = final_audio.subclip(0, video_clip.duration)
    
    # Set the audio to the video
    final_video = video_clip.set_audio(final_audio)
    
    # Write the final video to the output path
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

def get_file_dialogs():
    app = QApplication(sys.argv)
    
    # Ask the user to select the video file from the specified directory
    video_path, _ = QFileDialog.getOpenFileName(
        caption="Select the original video file",
        directory="src/components/temp/originalvideo",
        filter="Video Files (*.mp4 *.avi *.mov)"
    )
    
    # Ask the user to select the directory containing audio files
    audio_dir = QFileDialog.getExistingDirectory(
        caption="Select the directory containing audio files",
        directory="src/components/temp/tts_output"
    )
    
    # Define the output path for the final video
    output_path, _ = QFileDialog.getSaveFileName(
        parent=None,
        caption="Save Video File",
        directory="",
        filter="Video Files (*.mp4 *.avi);;All Files (*)"
    )
    
    return video_path, audio_dir, output_path

if __name__ == "__main__":
    video_path, audio_dir, output_path = get_file_dialogs()
    
    # Sync audio and video
    if video_path and audio_dir and output_path:
        sync_audio_video(video_path, audio_dir, output_path)
    else:
        print("Operation cancelled or invalid selection.")