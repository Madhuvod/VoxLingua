import sys
import os
import glob
from dotenv import load_env
# Add the parent directory of 'voice_conversion_models' to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Add the Applio directory to the Python path
applio_dir = os.path.join(parent_dir, 'voice_conversion_models', 'Applio')
sys.path.append(applio_dir)

# Now try to import from voice_conversion_models
from voice_conversion_models.Applio.rvc.infer.pipeline import Pipeline as VC
from voice_conversion_models.Applio.rvc.lib.utils import load_audio_infer, load_embedding
from voice_conversion_models.Applio.rvc.lib.algorithm.synthesizers import Synthesizer
from voice_conversion_models.Applio.rvc.configs.config import Config

import torch
import numpy as np
import librosa
import soundfile as sf
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model, Pipeline


TEMP_FOLDER = "temp"
ORIGINAL_AUDIO_FOLDER = os.path.join(TEMP_FOLDER, "originalaudio")
TRANSLATED_AUDIO_FOLDER = os.path.join(TEMP_FOLDER, "tts_output")
OUTPUT_FOLDER = os.path.join(TEMP_FOLDER, "clone_output")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_latest_file(folder_path, file_extension):
    """Get the most recently modified file with the given extension in the specified folder."""
    files = glob.glob(os.path.join(folder_path, f"*{file_extension}"))
    if not files:
        raise FileNotFoundError(f"No {file_extension} files found in {folder_path}")
    return max(files, key=os.path.getmtime)

# Get the paths dynamically
try:
    original_audio_path = get_latest_file(ORIGINAL_AUDIO_FOLDER, ".wav")
    translated_audio_path = get_latest_file(TRANSLATED_AUDIO_FOLDER, ".mp3")
    print(f"Original audio: {original_audio_path}")
    print(f"Translated audio: {translated_audio_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Handle the error appropriately (e.g., exit the script or use default values)

# Generate a unique output file name
output_file_name = f"cloned_{os.path.basename(original_audio_path)}"
output_path = os.path.join(OUTPUT_FOLDER, output_file_name)
print(f"Output will be saved to: {output_path}")

YOUR_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

if not YOUR_AUTH_TOKEN:
    raise ValueError("HF_AUTH_TOKEN not found in .env file")
# Load the pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=YOUR_AUTH_TOKEN)

# When using the diarization:
try:
    diarization_result = pipeline(original_audio_path)
    
    speaker_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments.append((turn.start, turn.end, speaker))
    
    print(f"Found {len(speaker_segments)} speaker segments")

except Exception as e:
    print(f"Error during diarization: {e}")
    # Handle the error appropriately (e.g., exit the script or use default values)
    speaker_segments = [(0, librosa.get_duration(filename=original_audio_path), "SPEAKER_1")]

def extract_audio_segment(audio_path, start_time, end_time):
    audio, sr = librosa.load(audio_path, sr=None, offset = start_time, duration=end_time-start_time)
    return audio, sr


device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_path = os.path.join(parent_dir, "rvc", "models", "pretraineds")

# Specify the exact subdirectory where your model is located
model_version = "pretrained_v2"  # or "pretrained_v1" or "pretraineds_custom"
model_path = os.path.join(base_model_path, model_version)

config_path = os.path.join(parent_dir, "rvc", "configs")

config = Config(config_path)

# Make sure the model file exists
model_files = glob.glob(os.path.join(model_path, "*.pth"))
if not model_files:
    raise FileNotFoundError(f"No model files found in {model_path}. Please download or generate the model files first.")

vc = VC(model_path, config, device=device)

def voice_conversion(source_audio, target_audio, sr):
    source_audio = load_audio_infer(source_audio, sr, config.resample_sr)
    target_audio = load_audio_infer(target_audio, sr, config.resample_sr)

    speaker_embedding = load_embedding(target_audio)
    converted_audio = vc.pipeline(source_audio, speaker_embedding = speaker_embedding, f0_up_key = 0)
    return converted_audio

translation_audio, sr = librosa.load(translated_audio_path, sr= None)

final_output = []
for start, end, speaker in speaker_segments:
    original_segment, original_sr = extract_audio_segment(original_audio_path, start, end)
    translated_segment = translation_audio[int(start*sr):int(end*sr)]
    converted_segment = voice_conversion(translated_segment, original_segment, sr)
    final_output.extend(converted_segment)


sf.write(output_path, np.concatenate(final_output), sr)
print(f"Voice conversion completed. Output saved to {output_path}")

