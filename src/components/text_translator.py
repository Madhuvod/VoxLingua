#text translator using MarianMT

import os
import json
from transformers import MarianMTModel, MarianTokenizer

TEMP_FOLDER = os.path.join('temp')


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
    try:
        full_path = os.path.join(TEMP_FOLDER, transcription_file)
        with open(full_path, "r", encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        original_text = transcription_data['text']
        source_language = transcription_data['language']

        model, tokenizer = load_model_tokenizer(source_language, target_language)
        
        translated_text = translated_text(original_text, model, tokenizer)

        translation_data = {
            "original_file": transcription_data.get('original_audio', 'unknown'),
            "source_language": source_language,
            "target_language": target_language,
            "original_text": original_text,
            "translated_text": translated_text
        }
        
        base_name = os.path.splitext(transcription_file)[0]
        translation_file = f"{base_name}_{target_language}_translation.json"
        with open(translation_file, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, ensure_ascii=False, indent=2)

        print(f"Translation completed and saved to {translation_file}")
        return translation_file
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
    # List available transcription files
    transcription_files = list_transcription_files()
    
    if not transcription_files:
        print("No transcription files found in the temp folder.")
    else:
        print("Available transcription files:")
        for i, file in enumerate(transcription_files, 1):
            print(f"{i}. {file}")
        
        # Get user selection
        selection = input("Enter the number of the file you want to translate: ")
        try:
            selected_file = transcription_files[int(selection) - 1]
            
            # Get target language from user
            target_language = input("Enter the target language code (e.g., 'es' for Spanish): ")
            
            # Process translation
            translation_file = process_translation(selected_file, target_language)
            print(f"Translation saved as: {translation_file}")
        except (ValueError, IndexError):
            print("Invalid selection. Please run the script again and enter a valid number.")




