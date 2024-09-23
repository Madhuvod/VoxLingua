import whisper
import os
import json

def transcribe_audio(audio_file_name):

    #audio path
    audio_file_path = os.path.join("temp", "originalaudio", audio_file_name)

    if not os.path.exists(audio_file_path):
        print(f"Error loading the file {audio_file_path}, could'nt find it")
        return None
    #model loading
    model = whisper.load_model("base")
    #transcribe
    result = model.transcribe(audio_file_path)

    transcription_data = {
        "original_audio": audio_file_name,
        "text" : result["text"],
        "language" : result["language"]
    }
    os.makedirs(os.path.join("temp", "transcriptions"), exist_ok=True)

    transcription_file_name = f"{os.path.splitext(audio_file_name)[0]}_transcription.json"
    json_file_path = os.path.join("temp","transcriptions" ,transcription_file_name)
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(transcription_data, f, ensure_ascii=False, indent=2)

    print(f"Transcription completed and saved to {json_file_path}")
    return transcription_file_name

if __name__ == "__main__":
    wav_files = [f for f in os.listdir(os.path.join("temp", "originalaudio")) if f.endswith(".wav")]
    if not wav_files:
        print("No .wav files found in the temp/transcriptions folder.")
    else:
        print("Available .wav files:")
        for i, file in enumerate(wav_files, 1):
            print(f"{i}. {file}")
        
        selection = input("Enter the number of the file you want to transcribe: ")
        try:
            selected_file = wav_files[int(selection) - 1]
            transcription_file = transcribe_audio(selected_file)
            if transcription_file:
                print(f"Transcription saved as: {transcription_file}")
        except (ValueError, IndexError):
            print("Invalid selection. Please run the script again and enter a valid number.")

