from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from pathlib import Path

import tensorflow as tf
from tfrecord_helper.tfrecord_helper import TFRecordHelper, TFRecordHelperWriter 


def create_tfrecord_from_image_list(imgs: Union[List[Path], str], 
                                    out_tfrecord_filepath: str = 'image.tfrecords', 
                                    height=416, 
                                    width=416,
                                    shard_size=10000) -> None:
  '''
  imgs: list of image paths (of type pathlib.Path or str) with assumptions that image actually exists at that path
  out_tfrecord_filepath: full path name of output tfrecord file

  return nothing. File written to
  '''
  parent = Path(out_tfrecord_filepath).parent
  stem = Path(out_tfrecord_filepath).stem
  extension = Path(out_tfrecord_filepath).suffix
  # out_tfrecord_filepath = str(out_tfrecord_filepath)

  imgs = [str(img) for img in imgs]

  # segment imgs in shards of 'shard_size' (default to 10000)
  imgs_shards = [imgs[i:i+shard_size] for i in range(0, len(imgs), shard_size)]
  n_imgs_shards = len(imgs_shards)

  def proc_img(x):   # mainly just resizing
    img_str = tf.io.read_file(x['filepath'])
    image = tf.image.decode_jpeg(img_str, channels=3)
    image = tf.image.resize(image, (height, width), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
    image = tf.cast(image, tf.uint8)
    resized_img_str = tf.io.encode_jpeg(image, quality=100)
    return resized_img_str

  for k, img_shard in enumerate(imgs_shards):
    if n_imgs_shards > 1:
      output_filename = str(parent/f'{stem}_{k:04d}_of_{n_imgs_shards:04d}{extension}')  # e.g. image_0000_of_0002.tfrecords
    else:
      output_filename = str(parent/f'{stem}{extension}')

    file_ds = tf.data.Dataset.from_tensor_slices(
        {
          'filename': [Path(f).name for f in img_shard],
          'filepath': img_shard
        }
    )

    data_ds = file_ds.map(lambda x: {'filename': x['filename'], 'image_raw': proc_img(x)}, num_parallel_calls=tf.data.AUTOTUNE)

    features = {
      'filename': TFRecordHelper.DataType.STRING,
      'image_raw': TFRecordHelper.DataType.STRING,   # bytes for the encoded jpeg, png, etc.
    }

    parse_fn = TFRecordHelper.parse_fn(features)

    with TFRecordHelperWriter(output_filename, features = features) as f:
      f.write(data_ds)