import yt_dlp
import os
import subprocess
import boto3

class VideoProcessor:
    def __init__(self, output_dir='temp', bucket_name='original-audio-video-voxlingua-main'):
        # Directory setup
        self.output_dir = output_dir
        self.audio_output_dir = os.path.join(output_dir, 'originalaudio')
        self.video_output_dir = os.path.join(output_dir, 'originalvideo')
        os.makedirs(self.audio_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # S3 client initialization
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_file_to_s3(self, local_file_path, s3_file_name):
        try:
            self.s3.upload_file(local_file_path, self.bucket_name, s3_file_name)
            print(f"Uploaded {s3_file_name} to {self.bucket_name} successfully.")
        except Exception as e:
            print(f"Error uploading {s3_file_name} to S3: {e}")

    def sanitize_filename(self, title):
        # Replacing spaces with underscores and remove invalid characters
        return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in title.replace(' ', '_'))

    def process_youtube_audio(self, url):
        try:
            print("Downloading audio...")
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'audio')
                sanitized_title = self.sanitize_filename(video_title)
                
                audio_path = os.path.join(self.audio_output_dir, f'{sanitized_title}.wav')
                # Rename the downloaded file to match the YouTube video title
                downloaded_filename = ydl.prepare_filename(info).replace('.webm', '.wav')
                os.rename(downloaded_filename, audio_path)

                print(f"Audio downloaded successfully as {audio_path}.")

                # Upload to S3
                s3_file_name = f"{sanitized_title}-audio.wav"
                self.upload_file_to_s3(audio_path, s3_file_name)

            return audio_path
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None

    def process_youtube_video(self, url):
        try:
            print("Downloading video...")
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'video')
                sanitized_title = self.sanitize_filename(video_title)

                video_path = os.path.join(self.video_output_dir, f'{sanitized_title}.mp4')
                # Rename the downloaded file to match the YouTube video title
                downloaded_filename = ydl.prepare_filename(info).replace('.webm', '.mp4')
                os.rename(downloaded_filename, video_path)

                print(f"Video downloaded successfully as {video_path}.")

                # Removing audio from the video
                video_no_audio_path = os.path.splitext(video_path)[0] + '_no_audio.mp4'
                command = [
                    'ffmpeg',
                    '-i', video_path,
                    '-c:v', 'copy',
                    '-an',
                    video_no_audio_path
                ]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    print("Video processed successfully (audio removed).")
                    os.remove(video_path)  # Remove the original video with audio

                    # Upload to S3
                    s3_file_name = f"{sanitized_title}-video_no_audio.mp4"
                    self.upload_file_to_s3(video_no_audio_path, s3_file_name)

                    return video_no_audio_path
                else:
                    print(f"FFmpeg failed: {result.stderr.decode('utf-8')}")
                    return None
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def process_youtube(self, url):
        audio_path = self.process_youtube_audio(url)
        video_path = self.process_youtube_video(url)
        return audio_path, video_path

# Example usage
if __name__ == "__main__":
    bucket_name = 'original-audio-video-voxlingua-main'  # Replace with your S3 bucket name
    processor = VideoProcessor(bucket_name=bucket_name)
    
    youtube_url = input("Enter the YouTube video URL: ")
    audio, video = processor.process_youtube(youtube_url)
    
    if audio:
        print(f"YouTube audio saved to: {audio}")
    else:
        print("Failed to process YouTube audio")
    
    if video:
        print(f"YouTube video (without audio) saved to: {video}")
    else:
        print("Failed to process YouTube video")
