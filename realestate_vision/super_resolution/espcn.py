from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union, Callable

import math
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers  # type: ignore

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img   # type: ignore


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL

def scaling(input_image):
  input_image = input_image / 255.0
  return input_image

# Processing tf.data.Dataset during training
def process_input(input, input_size, upscale_factor):
  input = tf.image.rgb_to_yuv(input)
  y, u, v = tf.split(input, 3, axis=-1)
  return tf.image.resize(y, [input_size, input_size], method='area')

def process_target(input):
  input = tf.image.rgb_to_yuv(input)
  y, u, v = tf.split(input, 3, axis=-1)
  return y

def create_model(upscale_factor=3, channels=1):
  conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
  inputs = keras.Input(shape=(None, None, channels))
  x = layers.Conv2D(64, 5, **conv_args)(inputs)
  x = layers.Conv2D(64, 3, **conv_args)(x)
  x = layers.Conv2D(32, 3, **conv_args)(x)
  x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
  outputs = tf.nn.depth_to_space(x, upscale_factor)

  return keras.Model(inputs, outputs)

# Utility functions
# * plot_results to plot an save an image.
# * get_lowres_image to convert an image to its low-resolution version.
# * upscale_image to turn a low-resolution image to a high-resolution version reconstructed by the model. 
# In this function, we use the y channel from the YUV color space as input to the model and then combine the output with the other channels to obtain an RGB image.

def plot_results(img, prefix, title):
  """Plot the result with zoom-in area."""
  img_array = img_to_array(img)
  img_array = img_array.astype("float32") / 255.0

  # Create a new figure with a default 111 subplot.
  fig, ax = plt.subplots()
  im = ax.imshow(img_array[::-1], origin="lower")

  plt.title(title)
  # zoom-factor: 2.0, location: upper-left
  axins = zoomed_inset_axes(ax, 2, loc=2)
  axins.imshow(img_array[::-1], origin="lower")

  # Specify the limits.
  x1, x2, y1, y2 = 200, 300, 100, 200
  # Apply the x-limits.
  axins.set_xlim(x1, x2)
  # Apply the y-limits.
  axins.set_ylim(y1, y2)

  plt.yticks(visible=False)
  plt.xticks(visible=False)

  # Make the line.
  mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
  plt.savefig(str(prefix) + "-" + title + ".png")
  plt.show()


def get_lowres_image(img, upscale_factor):
  """Return low-resolution image to use as model input."""
  return img.resize(
    (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
    PIL.Image.BICUBIC,
  )


def upscale_image(model, img):
  """Predict the result based on input image and restore the image as RGB."""
  ycbcr = img.convert("YCbCr")
  y, cb, cr = ycbcr.split()
  y = img_to_array(y)
  y = y.astype("float32") / 255.0

  input = np.expand_dims(y, axis=0)
  out = model.predict(input)

  out_img_y = out[0]
  out_img_y *= 255.0

  # Restore the image in RGB color space.
  out_img_y = out_img_y.clip(0, 255)
  out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
  out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
  out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
  out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
  out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
      "RGB"
  )
  return out_img


class ESPCNCallback(keras.callbacks.Callback):
  def __init__(self, test_img_paths: List, upscale_factor: int = 3):
    super(ESPCNCallback, self).__init__()
    self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

  # Store PSNR value in each epoch.
  def on_epoch_begin(self, epoch, logs=None):
    self.psnr = []

  def on_epoch_end(self, epoch, logs=None):
    print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
    if epoch % 20 == 0:
      prediction = upscale_image(self.model, self.test_img)
      plot_results(prediction, "epoch-" + str(epoch), "prediction")

  def on_test_batch_end(self, batch, logs=None):
    self.psnr.append(10 * math.log10(1 / logs["loss"]))


def configure_callbacks(checkpoint_filepath: str = None, upscale_factor: int = 3, test_img_paths: List = None) -> List[tf.keras.callbacks.Callback]:
  early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

  if checkpoint_filepath is None: checkpoint_filepath = "/tmp/checkpoint"

  model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor="loss",
      mode="min",
      save_best_only=True,
  )

  return [ESPCNCallback(test_img_paths=test_img_paths, upscale_factor=upscale_factor), 
          early_stopping_callback, 
          model_checkpoint_callback]


def get_create_compile_model(upscale_factor: int = 3, learning_rate: float = 0.001) -> keras.Model:
  model = create_model(upscale_factor=upscale_factor, channels=1)
  # model.summary()

  loss_fn = keras.losses.MeanSquaredError()
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(optimizer=optimizer, loss=loss_fn)

  return model

def train(model, train_ds, val_ds, epochs: int = 100, callbacks: List[tf.keras.callbacks.Callback] = None, checkpoint_filepath: str = None):
  if checkpoint_filepath is None: checkpoint_filepath = "checkpoint"

  history = model.fit(train_ds, 
                      epochs=epochs, 
                      callbacks=callbacks,
                      validation_data=val_ds,
                      verbose=2
                      )

  model.load_weights(checkpoint_filepath)

  return history


