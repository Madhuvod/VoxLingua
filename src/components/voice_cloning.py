import os
import sys
import torch
import torchaudio
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from dotenv import load_dotenv
import logging

from TTS.api import TTS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

load_dotenv()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

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


