import yt_dlp
import ffmpeg
import os

class VideoProcessor:
    def __init__(self, output_dir='temp'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_video(self, video_source):
        if video_source.startswith('http'):
            return self._process_youtube(video_source)
        else:
            return self._process_local(video_source)

    def _process_youtube(self, url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            audio_path = os.path.splitext(filename)[0] + '.wav'
        
        return audio_path

    def _process_local(self, file_path):
        filename = os.path.basename(file_path)
        audio_path = os.path.join(self.output_dir, os.path.splitext(filename)[0] + '.wav')
        
        (
            ffmpeg
            .input(file_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        
        return audio_path

# Example usage
if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Process YouTube video
    youtube_audio = processor.process_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(f"YouTube audio saved to: {youtube_audio}")
    
    # Process local video
    local_audio = processor.process_video("/path/to/local/video.mp4")
    print(f"Local video audio saved to: {local_audio}")
