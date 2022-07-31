from __future__ import print_function

import re

def get_listingId_from_image_name(image_name):
  # listingId_regex = re.compile(r'.*?(\d+).*.jpg$')
  listingId_regex = re.compile(r'.*?(\d{4,}).*.jpg$')    # listingId length > 4

  matches = listingId_regex.match(str(image_name))
  if matches is not None:
    listingId = matches.group(1)
    return listingId
  else:
    return "UNKNOWN"

def tf_dataset_peek(ds, loc, as_numpy=False):
  if as_numpy:
    ds = ds.as_numpy_iterator()
    
  for k, x in enumerate(ds):
    if k < loc: continue
    break

  return x

