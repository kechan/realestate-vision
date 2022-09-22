from __future__ import print_function

from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union

import re, pickle, gzip, hashlib
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model


def get_listingId_from_image_name(image_name):
  # listingId_regex = re.compile(r'.*?(\d+).*.jpg$')
  listingId_regex = re.compile(r'.*?(\d{4,}).*.jpg$')    # listingId length > 4

  matches = listingId_regex.match(str(image_name))
  if matches is not None:
    listingId = matches.group(1)
    return listingId
  else:
    return "UNKNOWN"

def get_listing_subfolder_from_listing_id(listing_id: str) -> str:
  ''' 
  Returns the subfolder of the listing_id. listing images can be stored in a hierarchy of subfolders.
  e.g. images for 12345678 can be stored under /12/34/56/78/
  '''
  folder = []
  for k in range(len(listing_id) // 2):
    i = int(listing_id[2*k: 2*k+2])
    folder.append(str(i))

  return '/'.join(folder)

def get_listing_folder_from_image_name(image_name: str) -> str:
  listingId = get_listingId_from_image_name(image_name)

  folder = []
  for k in range(len(listingId) // 2):
    i = int(listingId[2*k: 2*k+2])
    folder.append(str(i))

  return '/'.join(folder)  

def sha256digest(content, truncate_len=10):
  return hashlib.sha224(content.encode('utf-8')).hexdigest()[:truncate_len]          
        