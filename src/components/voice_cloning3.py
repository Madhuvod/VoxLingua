# import os
# import json
# import torch
# import torchaudio
# from pathlib import Path
# from GPT_SoVITS.TTS_infer_pack.TTS import TTS

# # Define paths
# BASE_DIR = Path(__file__).resolve().parent
# ORIGINAL_AUDIO_DIR = BASE_DIR / "temp" / "originalaudio"
# TRANSLATIONS_DIR = BASE_DIR / "temp" / "translations"
# OUTPUT_DIR = BASE_DIR / "temp" / "voice_cloned_output"
# MODEL_DIR = BASE_DIR.parent.parent / "voice_conversion_models" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"

# # Ensure output directory exists
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Load GPT-SoVITS model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tts = TTS(
#     gpt_path=MODEL_DIR / "s2G2333k.pth",  # Adjust based on your model
#     sovits_path=MODEL_DIR / "s2D2333k.pth",  # Adjust based on your model
#     device=device
# )

# def load_audio(file_path):
#     """Load audio file and return waveform."""
#     waveform, sample_rate = torchaudio.load(file_path)
#     return waveform.squeeze().numpy()

# def select_file(directory, extension):
#     """Allow user to select a file with a given extension."""
#     files = list(directory.glob(f"*{extension}"))
#     print(f"Available {extension} files:")
#     for i, file in enumerate(files):
#         print(f"{i + 1}: {file.name}")
#     choice = int(input(f"Select a file by number: ")) - 1
#     return files[choice]

# def train_sovits_with_original_audio(audio_file):
#     """Train SoVITS with the original audio to clone the speaker's voice."""
#     # Placeholder for training logic
#     print(f"Training SoVITS with {audio_file.name}...")

# def generate_translated_speech(audio_file, translation_file):
#     """Generate translated speech in the cloned voice."""
#     with open(translation_file, 'r') as f:
#         translation_data = json.load(f)
    
#     translated_text = translation_data.get("translated_text", "")
#     if not translated_text:
#         print(f"No translated text found in {translation_file.name}.")
#         return
    
#     reference_audio = load_audio(audio_file)
    
#     cloned_audio = tts.infer(
#         text=translated_text,
#         reference_audio=reference_audio,
#         emotion="Neutral",  # Adjust as needed
#         language="es",      # Adjust based on your target language
#     )
    
#     output_path = OUTPUT_DIR / f"cloned_{audio_file.stem}.wav"
#     torchaudio.save(output_path, torch.tensor(cloned_audio).unsqueeze(0), 24000)
#     print(f"Saved cloned audio to {output_path}")

# def sync_audio_with_video():
#     """Sync the generated audio with the original audio's timing."""
#     # Placeholder for diarization and syncing logic
#     print("Syncing audio with video...")

# def voice_clone_and_translate():
#     audio_file = select_file(ORIGINAL_AUDIO_DIR, ".wav")
#     translation_file = select_file(TRANSLATIONS_DIR, ".json")
    
#     train_sovits_with_original_audio(audio_file)
#     generate_translated_speech(audio_file, translation_file)
#     sync_audio_with_video()

# if __name__ == "__main__":
#     voice_clone_and_translate()