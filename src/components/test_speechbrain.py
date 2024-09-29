from speechbrain.inference import EncoderClassifier

print("SpeechBrain imported successfully!")

# Try to load a pre-trained model
spk_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

print("Pre-trained model loaded successfully!")