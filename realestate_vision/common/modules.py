# Commonly needed Deep Learning related modules

import tensorflow as tf
import numpy as np
import pandas as pd

AUTO = tf.data.AUTOTUNE

import PIL
# from PIL import Image

# from tensorflow.keras.models import Sequential, Model, load_model

# TODO: investigate proper python to import everything in this file

try:
  import matplotlib.pyplot as plt
  import seaborn as sns
except Exception as e:
  print(e)
  print("Not importing matplotlib and seaborn")
