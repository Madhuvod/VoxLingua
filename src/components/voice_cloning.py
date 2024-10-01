import os
import sys
import torch
import torchaudio
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