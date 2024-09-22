#text to speech using gTTs

import os
import json
from gtts import gTTS
import glob

TEMP_FOLDER = os.path.join('temp')
TRANSLATIONS_FOLDER = os.path.join(TEMP_FOLDER, 'translations')
TTS_OUTPUT_FOLDER = os.path.join(TEMP_FOLDER, 'tts_output')

# Ensure the TTS output folder exists
os.makedirs(TTS_OUTPUT_FOLDER, exist_ok=True)

translation_files = glob.glob(os.path.join(TRANSLATIONS_FOLDER, '*_translation.json'))

for translation_file in translation_files:
    with open(translation_file, 'r', encoding='utf-8') as file:
        translation_data = json.load(file)

    translated_text = translation_data.get('translated_text', '')
    target_language = translation_data.get('target_language', 'en')
    original_file = translation_data.get('original_file', 'unknown')

    tts = gTTS(text=translated_text, lang=target_language)
    
    # Generate the output filename
    output_filename = f"{os.path.splitext(original_file)[0]}_translated.mp3"
    output_path = os.path.join(TTS_OUTPUT_FOLDER, output_filename)
    
    # Save the audio file
    tts.save(output_path)
    
    print(f"Generated TTS file: {output_path}")

print("Text-to-speech conversion completed.")

 