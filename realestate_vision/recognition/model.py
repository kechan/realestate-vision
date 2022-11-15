from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union, Callable
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from realestate_core.common.run_config import home, bOnColab

import pandas as pd

class ExteriorClassifier:
  def __init__(self):
    self.model_files = [
                           'resnet50_distill_exteriors.acc.0.9106.h5',
                           'resnet50_distill_exteriors.acc.0.9114.h5',
                           'resnet50_distill_exteriors.acc.0.9131.h5',
                           'resnet50_distill_exteriors.acc.0.9116.h5',
                           'resnet50_distill_exteriors.acc.0.9072.h5'
                           ]

    self.labels = ['facade', 'backyard', 'view', 'exterior']

    self.img_height, self.img_width = 224, 224    # exterior classification model take in 224x224 images

  def predict(self, batch_img_ds: tf.data.Dataset, img_names: List[str] = None, data_src: str=None, return_pd_dataframe=False) -> pd.DataFrame:
    y_preds = []
    for m in self.model_files:
      print(m)
      model = load_model(home/'ListingImageClassification'/'training'/'exteriors'/m)
      input_img_height, input_img_width = model.input.shape[1:3]

      assert input_img_height == self.img_height and input_img_width == self.img_width, 'model input shape is not 224x224'
      
      y_pred = model.predict(batch_img_ds)
      y_pred = tf.nn.sigmoid(y_pred).numpy()     # distill model return logits
      y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    y_pred = np.mean(y_preds, axis=0)

    if not return_pd_dataframe: return y_pred
    else:
      df = self._convert_pred_to_df(y_pred, img_names, data_src)
      return df

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


