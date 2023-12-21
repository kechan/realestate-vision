from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union, Callable
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from realestate_core.common.run_config import home, bOnColab

import pandas as pd

class ExteriorClassifier:
  def __init__(self, model_dir=None, cache_all_models=False):
    self.model_files = [
      'resnet50_distill_exteriors.acc.0.9106.h5',
      'resnet50_distill_exteriors.acc.0.9114.h5',
      'resnet50_distill_exteriors.acc.0.9131.h5',
      'resnet50_distill_exteriors.acc.0.9116.h5',
      'resnet50_distill_exteriors.acc.0.9072.h5'
    ]
    self.cache_all_models = cache_all_models
    self.model_dir = model_dir
    if self.model_dir is None:
      self.model_dir = home/'ListingImageClassification'/'training'/'exteriors'

    if cache_all_models:
      self.models = [load_model(self.model_dir/m) for m in self.model_files]

    self.labels = ['facade', 'backyard', 'view', 'exterior']

    self.img_height, self.img_width = 224, 224    # exterior classification model take in 224x224 images

  def predict(self, 
              batch_img_ds: tf.data.Dataset = None, 
              img_array: np.ndarray = None,
              img_names: List[str] = None, 
              data_src: str=None, 
              return_pd_dataframe=False,
              verbose=1) -> Union[pd.DataFrame, np.ndarray]:

    # either img_array or batch_img_ds must be provided
    assert (img_array is not None) or (batch_img_ds is not None), 'Either img_array or batch_img_ds must be provided.'

    if img_array is not None:
      assert img_array.shape[1:] == (self.img_height, self.img_width, 3), f'img_array shape must be (N, {self.img_height}, {self.img_width}, 3)'

    inputs = img_array if img_array is not None else batch_img_ds 

    y_preds = []
    for i, m in enumerate(self.model_files):
      # print(m)
      if self.cache_all_models:
        model = self.models[i]
      else:
        model = load_model(home/'ListingImageClassification'/'training'/'exteriors'/m)

      input_img_height, input_img_width = model.input.shape[1:3]

      assert input_img_height == self.img_height and input_img_width == self.img_width, 'model input shape is not 224x224'
      
      y_pred = model.predict(inputs, verbose=verbose)
      y_pred = tf.nn.sigmoid(y_pred).numpy()     # distill model return logits
      y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    y_pred = np.mean(y_preds, axis=0)

    if not return_pd_dataframe: return y_pred
    else:
      df = self._convert_pred_to_df(y_pred, img_names, data_src)
      return df

  def predict_from_img_paths(self, img_paths: List[Path], batch_size=32, return_pd_dataframe=False, verbose=1) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(img_paths, str) or isinstance(img_paths, Path): img_paths = [img_paths]

    def read_decode_resize(fname):
      img = tf.image.decode_jpeg(tf.io.read_file(fname), channels=3)
      img = tf.image.resize(img, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
      return img

    img_names = [p.name for p in img_paths]
    img_paths = [str(p) for p in img_paths]

    file_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    img_ds = file_ds.map(read_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

    batch_img_ds = img_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return self.predict(batch_img_ds=batch_img_ds, img_names=img_names, return_pd_dataframe=return_pd_dataframe, verbose=verbose)


  def _convert_pred_to_df(self, y_pred: np.ndarray, img_names: List[str], data_src=None) -> pd.DataFrame:
    df = pd.DataFrame(data={
        'img': img_names,
        'p_facade': np.round(y_pred[:, 1].astype('float'), 4),    # first element was an indicator for hard vs. soft labels during training, should ignore during inference.
        'p_backyard': np.round(y_pred[:, 2].astype('float'), 4),
        'p_view': np.round(y_pred[:, 3].astype('float'), 4),
        'p_exterior': np.round(y_pred[:, 4].astype('float'), 4),
        'data_src': data_src
      })

    return df


class GeneralClassifier:
  INOUT_DOOR_CLASS_NAMES = ['indoor', 'other', 'outdoor']
  ROOM_CLASS_NAMES = ['basement', 'bathroom', 'bedroom', 'dining_room', 'garage', 'gym_room', 'kitchen', 'laundry_room', 'living_room', 'office', 'other', 'storage']
  BOOLEAN_FEATURE_CLASS_NAMES = ['fireplace', 'agpool', 'body_of_water', 'igpool', 'balcony', 'deck_patio_veranda', 'ss_kitchen', 'double_sink', 'upg_kitchen']

  def __init__(self, model_file=None):
    self.model_file = model_file
    if self.model_file is None:
      self.model_file = home/'ListingImageClassification'/'training'/'hydra_all'/'resnet50_hydra_all.acc.0.9322.h5'
    self.model = load_model(self.model_file, compile=False)

    self.img_height, self.img_width = 416, 416    # general classification model take in 416x416 images


  def predict(self,
              batch_img_ds: tf.data.Dataset = None, 
              img_array: np.ndarray = None,
              img_names: List[str] = None, 
              data_src: str=None, 
              return_pd_dataframe=False,
              verbose=1) -> Union[pd.DataFrame, np.ndarray]:

    # either img_array or batch_img_ds must be provided
    assert (img_array is not None) or (batch_img_ds is not None), 'Either img_array or batch_img_ds must be provided.'

    if img_array is not None:
      assert img_array.shape[1:] == (self.img_height, self.img_width, 3), f'img_array shape must be (N, {self.img_height}, {self.img_width}, 3)'

    inputs = img_array if img_array is not None else batch_img_ds

    # model = load_model(self.model_file, compile=False)
    input_img_height, input_img_width = self.model.input.shape[1:3]

    assert input_img_height == self.img_height and input_img_width == self.img_width, 'model input shape is not 416x416'
    
    yhats = self.model.predict(inputs, verbose=verbose)
    if not return_pd_dataframe: return yhats

    df = self._convert_pred_to_df(yhats, img_names, data_src)
    return df

  def predict_from_img_paths(self, img_paths: List[Path], batch_size=32, return_pd_dataframe=False, verbose=1) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(img_paths, str) or isinstance(img_paths, Path): img_paths = [img_paths]

    def read_decode_resize(fname):
      img = tf.image.decode_jpeg(tf.io.read_file(fname), channels=3)
      img = tf.image.resize(img, (416, 416), method=tf.image.ResizeMethod.BILINEAR)
      return img

    img_names = [p.name for p in img_paths]
    img_paths = [str(p) for p in img_paths]

    file_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    img_ds = file_ds.map(read_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

    batch_img_ds = img_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return self.predict(batch_img_ds=batch_img_ds, img_names=img_names, return_pd_dataframe=return_pd_dataframe, verbose=verbose)


  def _convert_pred_to_df(self, yhats: np.ndarray, img_names: List[str], data_src=None) -> pd.DataFrame:
    iodoor_yhats = yhats[0]
    room_yhats = yhats[1]

    fireplace_yhats = yhats[2]
    agpool_yhats = yhats[3]
    body_of_water_yhats = yhats[4]
    igpool_yhats = yhats[5]
    balcony_yhats = yhats[6]
    deck_patio_veranda_yhats = yhats[7]
    ss_kitchen_yhats = yhats[8]
    double_sink_yhats = yhats[9]
    upg_kitchen_yhats = yhats[10]

    room_top_3 = tf.math.top_k(room_yhats, k=3)
    inoutdoor = [GeneralClassifier.INOUT_DOOR_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(iodoor_yhats, axis=-1))]

    predictions_df = pd.DataFrame(data={
      # 'img': [Path(img).name for img in imgs],
      'img': img_names,
      'inoutdoor': inoutdoor,
      'p_iodoor': np.round(np.max(iodoor_yhats, axis=-1).astype('float'), 4),

      'room': [GeneralClassifier.ROOM_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(room_yhats, axis=-1))],
      'p_room': np.round(np.max(room_yhats, axis=-1).astype('float'), 4),

      'room_1': np.array(GeneralClassifier.ROOM_CLASS_NAMES)[room_top_3.indices.numpy()[:, 1]],
      'p_room_1': np.round(room_top_3.values.numpy()[:, 1].astype('float'), 4),
      'room_2': np.array(GeneralClassifier.ROOM_CLASS_NAMES)[room_top_3.indices.numpy()[:, 2]],
      'p_room_2': np.round(room_top_3.values.numpy()[:, 2].astype('float'), 4),

      'p_fireplace': np.round(np.squeeze(fireplace_yhats).astype('float'), 4),
      'p_agpool': np.round(np.squeeze(agpool_yhats).astype('float'), 4),
      'p_body_of_water': np.round(np.squeeze(body_of_water_yhats).astype('float'), 4),
      'p_igpool': np.round(np.squeeze(igpool_yhats).astype('float'), 4),
      'p_balcony': np.round(np.squeeze(balcony_yhats).astype('float'), 4),
      'p_deck_patio_veranda': np.round(np.squeeze(deck_patio_veranda_yhats).astype('float'), 4),
      'p_ss_kitchen': np.round(np.squeeze(ss_kitchen_yhats).astype('float'), 4),
      'p_double_sink': np.round(np.squeeze(double_sink_yhats).astype('float'), 4),
      'p_upg_kitchen': np.round(np.squeeze(upg_kitchen_yhats).astype('float'), 4),

      'data_src': data_src
    })

    return predictions_df



    

    