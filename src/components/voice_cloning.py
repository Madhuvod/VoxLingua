import sys
import os

# Add the Applio directory to Python path
applio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'voice_conversion_models', 'Applio'))
sys.path.append(applio_path)

# Import necessary modules from Applio
from vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid, AudioEncoder
import torch

# Your existing imports
import numpy as np
import librosa
import soundfile as sf
from pyannote.audio import Pipeline

# Paths
original_audio_path = "temp/originalaudio/original.wav"
translated_audio_path = "temp/translations/translated.wav"
output_path = "temp/output/final_output.wav"

# Load pyannote.audio pipeline
diarization = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Perform diarization
diarization_result = diarization(original_audio_path)

# Extract speaker segments
speaker_segments = []
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    speaker_segments.append((turn.start, turn.end, speaker))

# Function to extract audio segment
def extract_audio_segment(audio_path, start_time, end_time):
    audio, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=end_time-start_time)
    return audio, sr

# Set up Applio
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join(applio_path, "path/to/your/model.pth")
config_path = os.path.join(applio_path, "path/to/your/config.json")

# Initialize VC model
vc = VC(model_path, config_path, device=device)

# Function to perform voice conversion
def voice_conversion(source_audio, target_audio, sr):
    # Convert source_audio to the format expected by Applio
    source_audio = librosa.resample(source_audio, sr, vc.target_sample)
    
    # Perform voice conversion
    audio = vc.pipeline(source_audio, 
                        target_audio,  # You might need to adjust this depending on Applio's API
                        f0_up_key=0)
    
    return audio

# Load translated audio
translated_audio, sr = librosa.load(translated_audio_path, sr=None)

# Process each speaker segment
final_output = []
for start, end, speaker in speaker_segments:
    # Extract original speaker voice sample
    original_segment, _ = extract_audio_segment(original_audio_path, start, end)
    
    # Extract corresponding segment from translated audio
    translated_segment = translated_audio[int(start*sr):int(end*sr)]
    
    # Perform voice conversion
    converted_segment = voice_conversion(translated_segment, original_segment, sr)
    
    final_output.extend(converted_segment)

# Save the final output
sf.write(output_path, np.array(final_output), sr)

print(f"Voice conversion completed. Output saved to {output_path}")