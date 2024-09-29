import os
import torch
import torchaudio
import sys
import numpy as np
from pydub import AudioSegment
from speechbrain.inference import EncoderClassifier
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import torchaudio.transforms as T

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Add the OpenVoice directory to sys.path
openvoice_dir = os.path.join(parent_dir, 'voice_conversion_models', 'OpenVoice')
sys.path.append(openvoice_dir)

from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Print the sys.path for debugging
print("Python path:")
for path in sys.path:
    print(path)

# Define the path to the config file
config_path = os.path.join(parent_dir, 'voice_conversion_models', 'OpenVoice', 'checkpoints_v2', 'converter', 'config.json')

# Now try to import and initialize the classes
try:
    base_speaker_tts = BaseSpeakerTTS(config_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    tone_color_converter = ToneColorConverter(config_path, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("OpenVoice classes successfully imported and initialized!")
    
except Exception as e:
    print(f"An error occurred while initializing OpenVoice classes: {str(e)}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Function to perform diarization using an energy-based VAD
def diarize_audio(audio_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz (common for speech processing)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        print(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz")
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    
    # Normalize audio
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    
    # Simple energy-based VAD
    frame_length = int(0.025 * sample_rate)  # 25ms
    hop_length = int(0.010 * sample_rate)    # 10ms
    window = torch.hann_window(frame_length).to(waveform.device)
    
    # Compute Short-Time Fourier Transform (STFT)
    stft = torch.stft(
        waveform[0],
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        return_complex=True
    )
    
    # Compute energy
    energy = stft.abs().pow(2).sum(dim=1).sqrt()
    
    # Determine threshold
    threshold = energy.mean() * 0.5
    print(f"Energy threshold set to: {threshold.item()}")
    
    # Detect speech frames
    speech_frames = energy > threshold  # Tensor of shape [num_frames]
    
    # Initialize the speaker recognition model
    spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    
    # Extract embeddings for each speech segment
    embeddings = []
    segments = []
    i = 0
    len_speech_frames = speech_frames.shape[0]
    while i < len_speech_frames:
        if speech_frames[i]:
            start_frame = i
            # Accumulate frames until speech_frames[i] becomes False or we reach segment_length
            end_frame = i
            while end_frame < len_speech_frames and speech_frames[end_frame] and (end_frame - start_frame) * hop_length < sample_rate:
                end_frame += 1
            # Get start and end samples
            start_sample = start_frame * hop_length
            end_sample = min(start_sample + sample_rate, waveform.shape[1])  # 1 second segments
            segment = waveform[:, start_sample:end_sample]
            if segment.shape[1] >= int(0.5 * sample_rate):  # Minimum 0.5 sec
                try:
                    emb = spk_model.encode_batch(segment)
                    embeddings.append(emb.squeeze().cpu().numpy())
                    segments.append((start_sample / sample_rate, end_sample / sample_rate))
                except Exception as e:
                    print(f"Error encoding segment ({start_sample/sample_rate:.2f}s - {end_sample/sample_rate:.2f}s): {e}")
            i = end_frame
        else:
            i += 1
    
    if len(embeddings) == 0:
        print("No valid speech segments found after processing.")
        return {"segments": []}
    
    # Perform clustering on embeddings
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
    
    # Assign clusters to speech segments
    labels = kmeans.labels_
    
    # Group segments by speaker
    segments_grouped = []
    current_speaker = labels[0]
    current_start = segments[0][0]
    for idx in range(1, len(labels)):
        if labels[idx] != current_speaker:
            current_end = segments[idx-1][1]
            segments_grouped.append((current_start, current_end, current_speaker))
            current_speaker = labels[idx]
            current_start = segments[idx][0]
    # Add the last segment
    segments_grouped.append((current_start, segments[-1][1], current_speaker))
    
    return {"segments": segments_grouped}

# Function to extract speaker embeddings
def extract_speaker_embedding(audio_path):
    return tone_color_converter.extract_se(audio_path)

# Function to clone voice
def clone_voice(source_audio, target_audio, output_path):
    source_embedding = extract_speaker_embedding(source_audio)
    target_embedding = extract_speaker_embedding(target_audio)
    tone_color_converter.convert(target_audio, target_embedding, source_embedding, output_path)

# Function to list files and let user select one
def select_file(directory, file_type):
    files = sorted([f for f in os.listdir(directory) if f.endswith(file_type)])
    if not files:
        print(f"No {file_type} files found in {directory}")
        return None
    print(f"\nAvailable {file_type} files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    while True:
        try:
            choice = int(input(f"\nSelect a {file_type} file (1-{len(files)}): "))
            if 1 <= choice <= len(files):
                return os.path.join(directory, files[choice-1])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main process
def process_audio(original_audio, translated_audio, output_dir):
    print("Starting diarization...")
    diarization_result = diarize_audio(original_audio)
    if not diarization_result['segments']:
        print("Diarization returned no segments. Exiting.")
        sys.exit(1)
    
    print("Diarization complete. Extracting main speaker...")
    
    # Extract speakers and count occurrences
    speakers = [segment[2] for segment in diarization_result['segments']]
    unique_speakers, counts = np.unique(speakers, return_counts=True)
    main_speaker = unique_speakers[np.argmax(counts)]
    print(f"Main speaker identified as Speaker {main_speaker}")
    
    # Extract audio for main speaker
    main_speaker_audio = os.path.join(output_dir, "main_speaker.wav")
    audio, sample_rate = torchaudio.load(original_audio)
    main_speaker_segments = torch.zeros_like(audio)
    
    for segment in diarization_result['segments']:
        if segment[2] == main_speaker:
            start_sample = int(segment[0] * sample_rate)
            end_sample = int(segment[1] * sample_rate)
            main_speaker_segments[:, start_sample:end_sample] = audio[:, start_sample:end_sample]
    
    # Save the extracted main speaker audio
    torchaudio.save(main_speaker_audio, main_speaker_segments, sample_rate)
    print(f"Main speaker audio saved to: {main_speaker_audio}")
    
    print("Starting voice cloning...")
    output_path = os.path.join(output_dir, "cloned_translated_audio.wav")
    clone_voice(main_speaker_audio, translated_audio, output_path)
    print("Voice cloning complete.")
    
    return output_path

# Usage
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    original_audio_dir = os.path.join(parent_dir, "src", "components", "temp", "originalaudio")
    translated_audio_dir = os.path.join(parent_dir, "src", "components", "temp", "tts_output")
    output_dir = os.path.join(parent_dir, "src", "components", "temp", "voice_cloned_output")
    
    print(f"Original Audio Directory: {original_audio_dir}")
    print(f"Translated Audio Directory: {translated_audio_dir}")
    print(f"Output Directory: {output_dir}")
    
    original_audio = select_file(original_audio_dir, ".wav")
    if not original_audio:
        sys.exit(1)
    
    translated_audio = select_file(translated_audio_dir, ".mp3")
    if not translated_audio:
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured that the output directory exists: {output_dir}")
    
    cloned_audio_path = process_audio(original_audio, translated_audio, output_dir)
    print(f"Voice cloned audio saved to: {cloned_audio_path}")