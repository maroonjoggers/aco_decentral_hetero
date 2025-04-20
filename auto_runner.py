import os
import time
from utils import TRAINING_EPOCHS

for _ in range(TRAINING_EPOCHS):
    os.system("python main.py")
    print("Run completed. Restarting in 5 seconds...\n")
    time.sleep(5)