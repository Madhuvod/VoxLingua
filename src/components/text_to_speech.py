import os
import json
from gtts import gTTS

TEMP_FOLDER = os.path.join('temp')
TRANSLATIONS_FOLDER = os.path.join(TEMP_FOLDER, 'translations')
TTS_OUTPUT_FOLDER = os.path.join(TEMP_FOLDER, 'tts_output')

# Ensure the TTS output folder exists
os.makedirs(TTS_OUTPUT_FOLDER, exist_ok=True)

def text_to_speech(translation_file: str, target_language: str) -> str | None:
    """
    Convert text to speech using Google TTS with optimized settings.
    
    Args:
        translation_file: Path to the JSON file containing translation data
        target_language: Target language code for TTS
    
    Returns:
        str | None: Path to the generated audio file or None if error occurs
    """
    try:
        # Read file once and store in memory
        with open(translation_file, 'r', encoding='utf-8') as file:
            translation_data = json.load(file)

        translated_text = translation_data.get('translated_text', '')
        original_file = translation_data.get('original_file', 'unknown')

        # Create TTS with slower=False for faster processing
        tts = gTTS(text=translated_text, 
                  lang=target_language,
                  slow=False)  # Add this parameter
        
        output_filename = f"{os.path.splitext(os.path.basename(original_file))[0]}_translated.mp3"
        output_path = os.path.join(TTS_OUTPUT_FOLDER, output_filename)
        
        tts.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        return None

if __name__ == "__main__":
    import glob
    
    translation_files = glob.glob(os.path.join(TRANSLATIONS_FOLDER, '*_translation.json'))

    for translation_file in translation_files:
        target_language = json.load(open(translation_file, 'r', encoding='utf-8')).get('target_language', 'en')
        text_to_speech(translation_file, target_language)

    print("Text-to-speech conversion completed.")