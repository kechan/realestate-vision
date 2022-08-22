from __future__ import print_function

from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
import re, pickle, gzip, hashlib
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model


def isNone_or_NaN(x):
  return (x is None) or (isinstance(x, float) and np.isnan(x))

def get_listingId_from_image_name(image_name):
  # listingId_regex = re.compile(r'.*?(\d+).*.jpg$')
  listingId_regex = re.compile(r'.*?(\d{4,}).*.jpg$')    # listingId length > 4

  matches = listingId_regex.match(str(image_name))
  if matches is not None:
    listingId = matches.group(1)
    return listingId
  else:
    return "UNKNOWN"

def get_listing_folder_from_image_name(image_name: str) -> str:
  listingId = get_listingId_from_image_name(image_name)

  folder = []
  for k in range(len(listingId) // 2):
    i = int(listingId[2*k: 2*k+2])
    folder.append(str(i))

  return '/'.join(folder)  

def tf_dataset_peek(ds, loc, as_numpy=False):
  if as_numpy:
    ds = ds.as_numpy_iterator()
    
  for k, x in enumerate(ds):
    if k < loc: continue
    break

  return x

def load_from_pickle(filename, compressed=False):
  try:
    if not compressed:
      with open(str(filename), 'rb') as f:
        obj = pickle.load(f)
    else:
      with gzip.open(str(filename), 'rb') as f:
        obj = pickle.load(f)

  except Exception as ex:
    print(ex)
    return None

  return obj

def save_to_pickle(obj, filename, compressed=False):
  try:
    if not compressed:
      with open(str(filename), 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      with gzip.open(str(filename), 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

  except Exception as ex:
    print(ex)

def join_df(left, right, left_on, right_on=None, suffix='_y', how='left'):
    if right_on is None: right_on = left_on
    return left.merge(right, how=how, left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))


def sha256digest(content, truncate_len=10):
  return hashlib.sha224(content.encode('utf-8')).hexdigest()[:truncate_len]                     