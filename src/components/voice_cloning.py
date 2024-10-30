import os
import sys
import torch
import torchaudio
import numpy as np
from sklearn.cluster import KMeans
from speechbrain.inference.speaker import EncoderClassifier
from dotenv import load_dotenv
import logging



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

load_dotenv()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

tts_dir = os.path.join(parent_dir, 'voice_conversion_models', 'TTS')  # Adjust if necessary
sys.path.append(tts_dir)

from TTS.api import TTS
def initialize_tts():
    try:         
        tts_model_name = "tts_models/multilingual/multi-dataset/xtts-v2"
        tts = TTS(model_name = tts_model_name, progress_bar=False, gpu=torch.cuda.is_available())
        logging.info(f"Coqui TTS model {tts_model_name} initialized")
        return tts
    except Exception as e:
        logging.error(f"Failed to initialize TTS: {e}")
        sys.exit(1)

tts = initialize_tts()

def extract_embeddings(audio_path):
    try:
        embeddings = tts.embed_utterance(audio_path)
        logger.info(f"Embeddings extracted successfully from {audio_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to extract embeddings: {e}")
        return None

def extract_middle_segment(waveform, sample_rate, segment_duration=6.0): #hardcoded 6 seconds as of now, can be changed later
    total_duration = waveform.shape[1] / sample_rate
    if total_duration < segment_duration:
        logger.warning(f"Audio duration is less than the segment duration. Returning the entire audio.")
        return waveform
    mid_point = total_duration / 2
    start_time = mid_point - (segment_duration / 2)
    end_time = start_time + segment_duration

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time + sample_rate) #since audio data is processed in sample

    extracted_segment = waveform[:, start_sample:end_sample]
    logger.info(f"Extracted {segment_duration} seconds from the middle of the audio")
    return extracted_segment

def diarize_audio(audio_path, segment_duration=6.0):
    try: #loading of speaker recognition model
        spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir="pretrained_models/spkrec-ecapa-voxceleb")
        logger.info(f"Speaker recognition model Loaded!!")
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            weaveform = resampler(waveform)
            sample_rate = 16000
            logger.info(f"Resampled audio to {sample_rate}Hz")
            
        embeddings = spk_model.encode_batch(waveform) #extraction of embeddings from the audio
        num_embeddings = embeddings.shape[0]
        logger.info(f"No of embeddings extracted: {num_embeddings}")

        if num_embeddings == 1:
            logger.info(f"Only one speaker detected in the audio. Returning the entire audio.")
            labels = np.array([0])
            main_speaker = 0
        else:
            n_clusters = min(2, num_embeddings)
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(embeddings.cpu().numpy())
            labels = kmeans.labels_
            logger.info(f"KMeans clustering completed with {n_clusters} clusters")

            unique_labels, counts = np.unique(labels, return_counts=True)
            main_speaker = unique_labels[np.argmax(counts)]
            logger.info(f"Main speaker detected with label {main_speaker}")

            main_speaker_segments = [waveform[:, i] for i in range(embeddings.shape[0]) if labels[i] == main_speaker]
            extracted_audio = torch.cat(main_speaker_segments, dim=0).unsqueeze(0)

            desired_length = int(segment_duration * sample_rate)
        if extracted_audio.shape[1] > desired_length:
            mid_sample = desired_length // 2
            start = (extracted_audio.shape[1] // 2) - mid_sample
            end = start + desired_length
            extracted_audio = extracted_audio[:, start:end]
            logger.info(f"Truncated extracted audio to {segment_duration} seconds.")
        elif extracted_audio.shape[1] < desired_length:
            # Pad with silence if shorter
            pad_length = desired_length - extracted_audio.shape[1]
            padding = torch.zeros((extracted_audio.shape[0], pad_length))
            extracted_audio = torch.cat((extracted_audio, padding), dim=1)
            logger.info(f"Padded extracted audio to {segment_duration} seconds.")
        
        # Save the extracted main speaker audio
        extracted_audio_path = os.path.splitext(audio_path)[0] + "_main_speaker.wav"
        torchaudio.save(extracted_audio_path, extracted_audio, sample_rate)
        logger.info(f"Main speaker audio saved to: {extracted_audio_path}")
        
        return {"extracted_audio": extracted_audio_path}
    
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return None

def clone_voice(source_audio, target_text, output_path):
    """
    Synthesizes speech from the target text using the source speaker's embedding.
    
    Args:
        source_audio (str): Path to the source speaker's audio file.
        target_text (str): Translated text to synthesize.
        output_path (str): Path to save the synthesized audio.
    
    Returns:
        str or None: Path to the synthesized audio file or None if failed.
    """
    try:
        embedding = extract_embeddings(source_audio)
        if embedding is None:
            logger.error("Failed to extract speaker embedding.")
            return None
        
        # Synthesize speech with the extracted embedding
        tts.tts_to_file(
            text=target_text,
            speaker_embedding=embedding,
            file_path=output_path
        )
        logger.info(f"Voice cloned audio saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        return None

def process_audio(original_audio_path, translated_text_path, output_dir):
    try:
        # Perform diarization to extract main speaker's audio
        diarization_result = diarize_audio(original_audio_path)
        if not diarization_result:
            logger.warning("Diarization could not be performed due to insufficient data.")
            return None
        
        main_speaker_audio = diarization_result.get("extracted_audio")
        if not main_speaker_audio or not os.path.exists(main_speaker_audio):
            logger.warning("Failed to extract main speaker's audio.")
            return None
        
        # Read the translated text
        with open(translated_text_path, 'r', encoding='utf-8') as f:
            translated_text = f.read().strip()
            if not translated_text:
                logger.error("Translated text is empty.")
                return None
        
        # Define output path for the cloned audio
        original_audio_name = os.path.splitext(os.path.basename(original_audio_path))[0]
        output_filename = f"{original_audio_name}.cloned_translated_audio.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Perform voice cloning
        cloned_audio = clone_voice(main_speaker_audio, translated_text, output_path)
        if not cloned_audio:
            logger.error("Voice cloning process failed.")
            return None
        
        return cloned_audio
    
    except Exception as e:
        logger.error(f"Processing audio failed: {e}")
        return None


def select_file(directory, file_type):
    """
    Lists files of a given type in a directory and allows user to select one.
    
    Args:
        directory (str): Directory path.
        file_type (str): File extension to filter (e.g., '.wav', '.txt').
    
    Returns:
        str or None: Selected file path or None if no files found.
    """
    try:
        files = sorted([f for f in os.listdir(directory) if f.endswith(file_type)])
        if not files:
            logger.warning(f"No {file_type} files found in {directory}")
            return None
        logger.info(f"Available {file_type} files: {files}")
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
    except Exception as e:
        logger.error(f"Error selecting file: {e}")
        return None
    

if __name__ == "__main__":
    original_audio_dir = os.path.join(parent_dir, "src", "components", "temp", "originalaudio")
    translated_text_dir = os.path.join(parent_dir, "src", "components", "temp", "tts_output")
    output_dir = os.path.join(parent_dir, "src", "components", "temp", "voice_cloned_output")
    
    print(f"Original Audio Directory: {original_audio_dir}")
    print(f"Translated Text Directory: {translated_text_dir}")
    print(f"Output Directory: {output_dir}")
    
    original_audio = select_file(original_audio_dir, ".wav")
    if not original_audio:
        sys.exit(1)
    
    translated_text = select_file(translated_text_dir, ".txt")
    if not translated_text:
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Ensured that the output directory exists: {output_dir}")
    
    cloned_audio_path = process_audio(original_audio, translated_text, output_dir)
    if cloned_audio_path:
        print(f"Voice cloned audio saved to: {cloned_audio_path}")
    else:
        print("Voice cloning process failed.")