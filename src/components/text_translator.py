#text translator using MarianMT

import os
import json
from transformers import MarianMTModel, MarianTokenizer

TEMP_FOLDER = os.path.join('temp')
#TEMP_FOLDER = os.path.join('src', 'components', 'temp')
TRANSLATIONS_FOLDER = os.path.join(TEMP_FOLDER, 'translations')

# Ensure the translations folder exists
os.makedirs(TRANSLATIONS_FOLDER, exist_ok=True)

def load_model_tokenizer(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, max_length = 512, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens = True)
    return translated_text

def process_translation(transcription_file, target_language):
    try:
        full_path = os.path.join(TEMP_FOLDER, transcription_file)
        with open(full_path, "r", encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        original_text = transcription_data['text']
        source_language = transcription_data['language']
        
        model, tokenizer = load_model_tokenizer(source_language, target_language)
        if model is None:
            raise ValueError("Failed to load both specific and fallback models")
        
        translated = translate_text(original_text, model, tokenizer)
        if translated is None:
            raise ValueError("Translation failed")
        
        translation_data = {
            "original_file": transcription_data.get('original_audio', 'unknown'),
            "source_language": source_language,
            "target_language": target_language,
            "original_text": original_text,
            "translated_text": translated
        }
        
        base_name = os.path.splitext(os.path.basename(transcription_file))[0]
        translation_file = f"{base_name}_{target_language}_translation.json"
        full_translation_path = os.path.join(TRANSLATIONS_FOLDER, translation_file)
        
        with open(full_translation_path, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, ensure_ascii=False, indent=2)
        
        print(f"Translation completed and saved to {full_translation_path}")
        return full_translation_path
    except FileNotFoundError:
        print(f"Error: The file '{transcription_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{transcription_file}' is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None

def list_transcription_files():
    return [f for f in os.listdir(TEMP_FOLDER) if f.endswith("_transcription.json")]

if __name__ == "__main__":
    transcription_files = list_transcription_files()
    if not transcription_files:
        print("No transcription files found in the temp folder.")
    else:
        print("Available transcription files:")
        for i, file in enumerate(transcription_files, 1):
            print(f"{i}. {file}")
        
        while True:
            selection = input("Enter the number of the file you want to translate: ")
            try:
                selected_file = transcription_files[int(selection) - 1]
                target_language = input("Enter the target language code (e.g., 'es' for Spanish): ")
                translation_file = process_translation(selected_file, target_language)
                if translation_file:
                    print(f"Translation saved as: {translation_file}")
                else:
                    print("Translation failed. Please check the error messages above.")
                break
            except (ValueError, IndexError):
                print("Invalid selection. Please enter a valid number.")