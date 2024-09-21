import yt_dlp
import os

class VideoProcessor:
    def __init__(self, output_dir='temp'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_youtube(self, url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                audio_path = os.path.splitext(filename)[0] + '.wav'
            return audio_path
        except Exception as e:
            print(f"Error processing YouTube video: {e}")
            return None

# Example usage
if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Process YouTube video
    youtube_url = input("Enter the YouTube video URL: ")
    youtube_audio = processor.process_youtube(youtube_url)
    if youtube_audio:
        print(f"YouTube audio saved to: {youtube_audio}")
    else:
        print("Failed to process YouTube video")