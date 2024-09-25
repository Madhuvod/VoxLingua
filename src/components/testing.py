import sys
import os
import glob

applio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'voice_conversion_models', 'Applio'))
sys.path.append(applio_path)
print(f"Applio path: {applio_path}")