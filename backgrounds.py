import pickle
import random

from matplotlib import pyplot as plt

data_dir = "data"  # Directory that will contain all kinds of data (the data we download and the data we generate)

# Pickle file containing the background images from the DTD
backgrounds_pck_fn = data_dir + "/backgrounds.pck"

# Pickle file containing the card images
cards_pck_fn = data_dir + "/cards.pck"

