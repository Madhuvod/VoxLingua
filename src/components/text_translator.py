#text translator using MarianMT

import os
import json
from transformers import MarianMTModel, MarianTokenizer

TEMP_FOLDER = os.path.join('src', 'components', 'temp')


def load_model_tokenizer(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

def translated_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, max_length = 512, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens = True)
    return translated_text

def process_translation(transcription_file, target_language):
    transcription_path = os.path.join(TEMP_FOLDER, transcription_file)

    with open(transcription_path, "r", encoding='utf-8') as f:
        transcription_data = json.load(f)
    

