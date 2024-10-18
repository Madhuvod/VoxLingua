import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


tts.tts_to_file(text="Mucha gente dijo asiático, LOL, pensé que eras asiático. No sé si es porque tengo muchos amigos asiáticos. O si es como mi delineador de ojos alado, que como todo el mundo usa. Pero la gente ha estado asumiendo que soy asiático durante años. Tanto es así que la gente en la televisión offline, me llaman un asiático honorífico. Pero en realidad, soy orgullosamente marroquí. Siento que podrías golpear el culo de alguien si fuera necesario. Y sí, quiero decir, ¿no puede alguien? Pensé que era normal. Como, si necesitaba para la autodefensa, voy a salir todo para mí siendo como la supervivencia el más apto. Estoy haciendo lo que sea necesario hacer. También, si estás molestando a uno de mis amigos, se acabó.", speaker_wav="/kaggle/input/pokimane_target/other/default/1/pokimane.wav", language="es", file_path="output.wav")