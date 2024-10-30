# Voxlingua

**Voxlingua** is a project that allows users to input a YouTube video link, choose a target language, and receive the original video with translated audio while preserving the original speaker's voice. This project employs multiple technologies to process video, recognize speech, translate text, convert text to speech, clone voices, and synchronize audio with video.

## Project Status

decided to use coquiXTTS w ec2 instance for the voice cloning part, f5-tts is also a better alternative but pnly supports eng and chinese. integrating aws s3 and ec2, lambda with the project, done with video_processing.py

## Project Architecture

The workflow of Voxlingua consists of six primary steps:

1. **Video Processing**:
   - The user uploads a YouTube link, and the video is processed using `yt-dlp` for download and `ffmpeg` for extraction of the audio stream.

2. **Speech Recognition**:
   - The extracted audio is processed through **OpenAI's Whisper** for speech recognition, converting the speech in the original language to text.

3. **Text Translation**:
   - The recognized text is translated into the target language using the **MarianMT Model from Hugging Face Transformers**.

4. **Text-to-Speech**:
   - The translated text is converted into speech using **Google Text-to-Speech (gTTS)**, generating an audio file in the target language.

5. **Voice Cloning**:
   - Using **GPT-SoVITS / OpenVoice**, the generated audio is transformed to retain the original speaker's voice, ensuring that the translated audio mimics the pitch and tone of the speaker in the original video.

6. **Audio-Video Sync**:
   - Finally, the translated audio is synced back to the video using `ffmpeg`, producing a video with the translated audio but preserving the speaker's original voice characteristics.

## Installation

To set up and run Voxlingua locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Voxlingua.git
   cd Voxlingua
   ```

## Basic Architecture

The basic architecture v1 is illustrated below:

![Architecture](image-1.png)

## Key Technologies

- **yt-dlp**: Used to download the video and extract the audio.
- **ffmpeg**: For video processing and synchronization of audio with video.
- **Whisper (OpenAI)**: For speech-to-text conversion.
- **Hugging Face Transformers**: For text translation into the desired language.
- **Google Text-to-Speech (gTTS)**: For text-to-speech generation in the target language.
- **GPT-SoVITS**: For cloning the speaker’s voice to retain their unique vocal characteristics.
- **Gradio**: For creating an interactive user interface.

## Usage

1. **Input a YouTube Video:**
   - Enter the YouTube URL in the input field.
   - Select the target language for translation.

2. **Process and Output:**
   - Voxlingua will process the video in the background and provide a downloadable link for the output video with translated audio in the original speaker’s voice.

## Future Improvements

- implement the trained voice cloning model for accurate voice cloning
- Add more language support for translation.
- Improve real-time performance of voice cloning.
- Enable additional customization options for the user.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## License

This project is licensed under the MIT License.