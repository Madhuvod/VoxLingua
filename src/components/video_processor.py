import yt_dlp
import os
import subprocess

class VideoProcessor:
    def __init__(self, output_dir='temp'):
        # Directory setup
        self.audio_output_dir = os.path.join(output_dir, 'originalaudio')
        self.video_output_dir = os.path.join(output_dir, 'originalvideo')
        os.makedirs(self.audio_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)

    def process_youtube_audio(self, url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': os.path.join(self.audio_output_dir, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        try:
            print("Downloading audio...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                audio_path = os.path.splitext(filename)[0] + '.wav'
            print(f"Audio downloaded successfully.")
            return audio_path
        except Exception as e:
            print(f"Error downloading audio: {e}")
        return None

    def process_youtube_video(self, url):
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
            'outtmpl': os.path.join(self.video_output_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegVideoRemuxer',
                'preferedformat': 'mp4',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        try:
            print("Downloading video...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                video_path = os.path.splitext(filename)[0] + '.mp4'
            
            print("Removing audio from the video...")
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
                print("Video processed successfully.")
                os.remove(video_path)
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
    processor = VideoProcessor()
    
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