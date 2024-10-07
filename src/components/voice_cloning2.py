import os
import torch
import torchaudio
import sys
import numpy as np
import json
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
def diarize_audio(audio_path, segment_duration=6.0):
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Initialize the speaker recognition model
        spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        
        # Extract embeddings
        embeddings = spk_model.encode_batch(waveform)
        num_embeddings = embeddings.shape[0]
        
        if num_embeddings == 1:
            labels = np.array([0])
            main_speaker = 0
            extracted_audio = waveform
        else:
            n_clusters = min(2, num_embeddings)
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(embeddings.cpu().numpy())
            labels = kmeans.labels_
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            main_speaker = unique_labels[np.argmax(counts)]
            
            main_speaker_segments = [waveform[:, i] for i in range(embeddings.shape[0]) if labels[i] == main_speaker]
            extracted_audio = torch.cat(main_speaker_segments, dim=1).unsqueeze(0)
            
            desired_length = int(segment_duration * sample_rate)
            if extracted_audio.shape[1] > desired_length:
                mid_sample = desired_length // 2
                start = (extracted_audio.shape[1] // 2) - mid_sample
                end = start + desired_length
                extracted_audio = extracted_audio[:, start:end]
            elif extracted_audio.shape[1] < desired_length:
                pad_length = desired_length - extracted_audio.shape[1]
                padding = torch.zeros((extracted_audio.shape[0], pad_length))
                extracted_audio = torch.cat((extracted_audio, padding), dim=1)
        
        # Save the extracted main speaker audio
        extracted_audio_path = os.path.splitext(audio_path)[0] + "_main_speaker.wav"
        torchaudio.save(extracted_audio_path, extracted_audio, sample_rate)
        
        # Return the extracted audio path and segments
        return {"extracted_audio": extracted_audio_path, "segments": [(0, extracted_audio.shape[1] / sample_rate, main_speaker)]}
    
    except Exception as e:
        print(f"Diarization failed: {e}")
        return None

# Function to extract speaker embeddings
def extract_speaker_embedding(audio_path):
    return tone_color_converter.extract_se(audio_path)

# Function to clone voice using translated text
def clone_voice_with_text(source_audio, translated_text, output_path):
    try:
        # Extract speaker embedding from the source audio
        source_embedding = extract_speaker_embedding(source_audio)
        
        # Assuming the speaker embedding can be used as a speaker ID
        # You might need to map the embedding to a speaker ID if required
        speaker_id = source_embedding  # Adjust this line as needed
        
        # Use the tts method to synthesize the audio
        base_speaker_tts.tts(
            text=translated_text,
            output_path=output_path,
            speaker=speaker_id,  # Ensure this is the correct format for speaker
            language='English',  # Adjust language if needed
            speed=1.0
        )
        
        print(f"Voice cloned audio saved to: {output_path}")
    except Exception as e:
        print(f"Voice cloning failed: {e}")

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

# Function to read translated text from a JSON file
def read_translated_text(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("translated_text", "")
    except Exception as e:
        print(f"Failed to read translated text from JSON: {e}")
        return ""

# Main process
def process_audio(original_audio, translated_json_path, output_dir):
    print("Starting diarization...")
    diarization_result = diarize_audio(original_audio)
    if not diarization_result['segments']:
        print("Diarization returned no segments. Exiting.")
        return None
    
    print("Diarization complete. Extracting main speaker...")
    
    # Extract main speaker audio
    main_speaker_audio = diarization_result['extracted_audio']
    
    # Read the translated text from JSON
    translated_text = read_translated_text(translated_json_path)
    if not translated_text:
        print("No translated text found in JSON. Exiting.")
        return None
    
    # Define output path for the cloned audio
    original_audio_name = os.path.splitext(os.path.basename(original_audio))[0]
    output_filename = f"{original_audio_name}.cloned_translated_audio.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    # Perform voice cloning
    clone_voice_with_text(main_speaker_audio, translated_text, output_path)
    
    return output_path

# Usage
if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    original_audio_dir = os.path.join(parent_dir, "src", "components", "temp", "originalaudio")
    translated_json_dir = os.path.join(parent_dir, "src", "components", "temp", "translations")
    output_dir = os.path.join(parent_dir, "src", "components", "temp", "voice_cloned_output")
    
    print(f"Original Audio Directory: {original_audio_dir}")
    print(f"Translated JSON Directory: {translated_json_dir}")
    print(f"Output Directory: {output_dir}")
    
    original_audio = select_file(original_audio_dir, ".wav")
    if not original_audio:
        sys.exit(1)
    
    translated_json_path = select_file(translated_json_dir, ".json")
    if not translated_json_path:
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured that the output directory exists: {output_dir}")
    
    cloned_audio_path = process_audio(original_audio, translated_json_path, output_dir)
    if cloned_audio_path:
        print(f"Voice cloned audio saved to: {cloned_audio_path}")
    else:
        print("Voice cloning process failed.")