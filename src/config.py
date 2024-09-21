import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Temporary folder for file processing
TEMP_FOLDER = os.path.join(BASE_DIR, 'src', 'components', 'temp')