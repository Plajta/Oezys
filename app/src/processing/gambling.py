import random
import os

# Get random AFM file from directory ../../../data/raw/diabetes without format .bmp at the end

def get_random_afm_file():
    afm_file = random.choice([f for f in os.listdir("../../../data/raw/diabetes") if f.endswith(".bmp")])
    return os.path.join("../../../data/raw/diabetes", afm_file)


# Normalizing


# Go through


# 