from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from tfrecord_helper.tfrecord_helper import TFRecordHelper, TFRecordHelperWriter
from realestate_core.common.run_config import home, bOnColab
from realestate_core.common.utils import join_df
from realestate_vision.common.utils import get_listingId_from_image_name

def gen_inference_for(ds: tf.data.Dataset, tuple_pos_img: int = 0, tuple_pos_filename: int = 1) -> pd.DataFrame:
  '''
  Use hydra and exterior models to generate inference given a tf.data.Dataset whose example is a tuple of (image, filename)
  image is at least 416x416x3
  
  As part of ML workflow, or certain context, these predictions/inferences are interpreted as weak or soft labels
  '''

  for x in ds.take(1): break
  n_args = len(x)    # number of arguments in the tuple per example

  def get_filename(*x):
    return x[tuple_pos_filename]

  imgs = [f.numpy().decode('utf-8') for f in ds.map(get_filename, num_parallel_calls=tf.data.AUTOTUNE)]

  batch_size = 32
  batch_ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  INOUT_DOOR_CLASS_NAMES = ['indoor', 'other', 'outdoor']
  ROOM_CLASS_NAMES = ['basement', 'bathroom', 'bedroom', 'dining_room', 'garage', 'gym_room', 'kitchen', 'laundry_room', 'living_room', 'office', 'other', 'storage']
  BOOLEAN_FEATURE_CLASS_NAMES = ['fireplace', 'agpool', 'body_of_water', 'igpool', 'balcony', 'deck_patio_veranda', 'ss_kitchen', 'double_sink', 'upg_kitchen']

  model = load_model(home/'ListingImageClassification'/'training'/'hydra_all'/'resnet50_hydra_all.acc.0.9322.h5', compile=False)

  yhats = model.predict(batch_ds)

  iodoor_yhats, room_yhats, fireplace_yhats, agpool_yhats, body_of_water_yhats, igpool_yhats, balcony_yhats, deck_patio_veranda_yhats, ss_kitchen_yhats, double_sink_yhats, upg_kitchen_yhats = yhats
  room_top_3 = tf.math.top_k(room_yhats, k=3)

  inoutdoor = [INOUT_DOOR_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(iodoor_yhats, axis=-1))]

  predictions_df = pd.DataFrame(data={
    'img': imgs,
    'inoutdoor': inoutdoor,
    'p_iodoor': np.round(np.max(iodoor_yhats, axis=-1).astype('float'), 4),

    'room': [ROOM_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(room_yhats, axis=-1))],
    'p_room': np.round(np.max(room_yhats, axis=-1).astype('float'), 4),

    'room_1': np.array(ROOM_CLASS_NAMES)[room_top_3.indices.numpy()[:, 1]],
    'p_room_1': np.round(room_top_3.values.numpy()[:, 1].astype('float'), 4),
    'room_2': np.array(ROOM_CLASS_NAMES)[room_top_3.indices.numpy()[:, 2]],
    'p_room_2': np.round(room_top_3.values.numpy()[:, 2].astype('float'), 4),

    'p_fireplace': np.round(np.squeeze(fireplace_yhats).astype('float'), 4),
    'p_agpool': np.round(np.squeeze(agpool_yhats).astype('float'), 4),
    'p_body_of_water': np.round(np.squeeze(body_of_water_yhats).astype('float'), 4),
    'p_igpool': np.round(np.squeeze(igpool_yhats).astype('float'), 4),
    'p_balcony': np.round(np.squeeze(balcony_yhats).astype('float'), 4),
    'p_deck_patio_veranda': np.round(np.squeeze(deck_patio_veranda_yhats).astype('float'), 4),
    'p_ss_kitchen': np.round(np.squeeze(ss_kitchen_yhats).astype('float'), 4),
    'p_double_sink': np.round(np.squeeze(double_sink_yhats).astype('float'), 4),
    'p_upg_kitchen': np.round(np.squeeze(upg_kitchen_yhats).astype('float'), 4)
  })

  # predictions for exterior types, use ensemble of models
  exteriors_model_files = [
                           'resnet50_distill_exteriors.acc.0.9106.h5',
                           'resnet50_distill_exteriors.acc.0.9114.h5',
                           'resnet50_distill_exteriors.acc.0.9131.h5',
                           'resnet50_distill_exteriors.acc.0.9116.h5',
                           'resnet50_distill_exteriors.acc.0.9072.h5'
                           ]

  exteriors_labels = ['facade', 'backyard', 'view', 'exterior']

  img_height, img_width = 224, 224    # exterior classification model take in 224x224 images

  if n_args == 2:
    def resize_jpg(img, filename):
      img = tf.image.resize(img, (img_height, img_width), method=tf.image.ResizeMethod.BILINEAR)
      return img, filename
  elif n_args == 3:
    def resize_jpg(img, filename, r):
      img = tf.image.resize(img, (img_height, img_width), method=tf.image.ResizeMethod.BILINEAR)
      return img, filename, r
  else:
    raise ValueError('ds must have 2 or 3 arguments per example')

  ds = ds.map(resize_jpg, num_parallel_calls=tf.data.AUTOTUNE)
  batch_ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  y_preds = []
  for m in exteriors_model_files:
    print(m)
    model = load_model(home/'ListingImageClassification'/'training'/'exteriors'/m)
    input_img_height, input_img_width = model.input.shape[1:3]

    assert input_img_height == img_height and input_img_width == img_width, 'model input shape is not 224x224'
    
    y_pred = model.predict(batch_ds)
    y_pred = tf.nn.sigmoid(y_pred).numpy()     # distill model return logits
    y_preds.append(y_pred)

  y_preds = np.stack(y_preds)
  y_pred = np.mean(y_preds, axis=0)

  exterior_predictions_df = pd.DataFrame(data={
    'img': imgs,
    'p_facade': np.round(y_pred[:, 1].astype('float'), 4),    # first element was an indicator for hard vs. soft labels during training, should ignore during inference.
    'p_backyard': np.round(y_pred[:, 2].astype('float'), 4),
    'p_view': np.round(y_pred[:, 3].astype('float'), 4),
    'p_exterior': np.round(y_pred[:, 4].astype('float'), 4)
  })

  # combine both set of predictions
  predictions_df = join_df(predictions_df, exterior_predictions_df, left_on='img', how='inner')

  # remove irrelevant exterior predictions from non outdoor 
  idx = predictions_df.q_py("inoutdoor.isin(['indoor', 'other'])").index
  predictions_df.loc[idx, 'p_facade'] = np.NaN
  predictions_df.loc[idx, 'p_backyard'] = np.NaN
  predictions_df.loc[idx, 'p_view'] = np.NaN
  predictions_df.loc[idx, 'p_exterior'] = np.NaN

  predictions_df['listingId'] = predictions_df.img.apply(get_listingId_from_image_name)

  return predictions_df
