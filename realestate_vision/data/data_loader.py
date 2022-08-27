from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union

from pathlib import Path
from realestate_nlp.common.run_config import home, bOnColab
from realestate_core.common.utils import isNone_or_NaN

from ..common.utils import sha256digest
from ..common.modules import AUTO

import tensorflow as tf
import pandas as pd
import numpy as np

import sqlalchemy as db
from sqlalchemy import Table, Column, Integer, String, Text, Boolean, Float, MetaData, BigInteger
from sqlalchemy import text, select
from sqlalchemy import inspect

from tfrecord_helper.tfrecord_helper import TFRecordHelper, TFRecordHelperWriter

BATCH_SIZE = 100   # want this to be a perfect square 
SQRT_BATCH_SIZE = int(np.sqrt(BATCH_SIZE))

class DataLoader:
  def __init__(self):
    self.home = home
    self.bOnColab = bOnColab
    
    db_path = home/'ListingImagesData'/'metadata'/'images.db'

    try:
      self.engine = db.create_engine(f"sqlite:///{db_path}", echo=False)

      self.inspector = inspect(self.engine)
      schema = self.inspector.get_schema_names()[0]

      self.db_conn = self.engine.connect()
      # print(self.inspector.get_table_names())
      self._setup_db_metadata()

      print("Connected to image database")
    except:
      print("Error connecting to image database")
      self.db_conn = None
      raise

  def _setup_db_metadata(self):
    meta = MetaData()

    self.image_labels = Table(
        'image_labels', meta,
        Column('id', Integer, primary_key=True),
        Column('name', String),
        Column('human_inoutdoor', String),
        Column('human_room', String),
        Column('human_facade', Boolean),
        Column('human_backyard', Boolean),
        Column('human_view', Boolean),
        Column('human_exterior', Boolean),
        Column('human_fireplace', Boolean),
        Column('human_igpool', Boolean),
        Column('human_agpool', Boolean),
        Column('human_body_of_water', Boolean),
        Column('human_ss_kitchen', Boolean),
        Column('human_double_sink', Boolean),
        Column('human_upg_kitchen', Boolean),
        Column('human_deck_patio_veranda', Boolean),
        Column('human_balcony', Boolean),
        Column('human_veranda', Boolean),
        Column('human_hw_floor', Boolean),
        Column('human_open_concept', Boolean),
        Column('human_deck_patio', Boolean),
        Column('human_upg_counter', Boolean),
        Column('human_gourmet_kitchen', Boolean),
        Column('annotation', Text)
    )
    if 'image_labels' not in self.inspector.get_table_names():
      meta.create_all(self.engine)

    self.images = Table(
      'images', 
      meta,
      Column('id', Integer, primary_key=True),
      Column('name', String),
      Column('origin', String),     
      Column('img_height', Integer), 
      Column('img_width', Integer), 
      Column('orig_aspect_ratio', Float),
      Column('hash', BigInteger),
    )
    if 'images' not in self.inspector.get_table_names():
      meta.create_all(self.engine)

  # from image db
  def get_image_labels_df(self) -> pd.DataFrame:
    if self.db_conn is None:
      return None

    df = pd.read_sql("select * from image_labels", self.db_conn, index_col='id')

    boolean_cols = [c.name for c in self.image_labels.columns if str(c.type) == 'BOOLEAN']
    for c in boolean_cols:
      df[c] = df[c].apply(lambda x: np.NaN if isNone_or_NaN(x) else x == 1)

    return df

  def get_images_df(self) -> pd.DataFrame:
    if self.db_conn is None:
      return None

    df = pd.read_sql("select * from images", self.db_conn, index_col='id')

    return df

  def get_image_info(self, name: str) -> pd.DataFrame:
    '''
    Obtain metadata for a single given image
    '''
    if self.db_conn is None:
      return None

    # ensure CREATE UNIQUE INDEX idx_images_name ON images (name);
    df = pd.read_sql(f"select * from images where name = '{name}'", self.db_conn, index_col='id')

    return df

  def get_image_as_tensor(self, name: str) -> tf.Tensor:
    if self.db_conn is None:
      return None

    df = self.get_image_info(name)
    if df is None:
      return None

    origin = df['origin'].values[0]
    if origin.startswith('RLP_listing'):
      tfrecord_filename = origin.split(' ')[-1]
    else:
      tfrecord_filename = origin
    
    # img = tf.io.read_file(img_path)
    # img = tf.image.decode_jpeg(img, channels=3)

    ds = self.get_big_archive_tfrecords_ds(tfrecord_filename)

    # Linear scan ds till it hits, this is highly inefficient, but it works for now
    for img, fname, r in ds:
      filename = fname.numpy().decode('utf-8')
      if filename == name:
        return img
    
  # Big archive (original images available on external hard drive)
  def get_big_archive_tfrecords_ds(self, tfrecord_filename: str = None, return_decode_image: bool = True) -> tf.data.Dataset:
    '''
    TFRecord tf.data.Dataset from massive set of listing images stored/archived originally on external hard drives
    Naming format and location 
         home/'ListingImageClassification'/'working_dir'/'tfrecords'/'listing_images_shard_*.tfrecords'
         home/'ListingImageClassification'/'working_dir'/'tfrecords'/'SamsungT5_listing_images_shard_*.tfrecords'

    Note: some set of tfrecords are on google cloud. 

    TODO: fix to account for tfrecords stored on google cloud.
    '''

    feature_desc = {
      'filename': TFRecordHelper.DataType.STRING,
      'image_raw': TFRecordHelper.DataType.STRING,
      'orig_aspect_ratios': TFRecordHelper.DataType.FLOAT
    }
    parse_fn = TFRecordHelper.parse_fn(feature_desc=feature_desc)

    if tfrecord_filename is None:
      # tfrecords set one: (original images not on samsung t5)
      tfrecords = (home/'ListingImageClassification'/'working_dir'/'tfrecords').lf('*.tfrecords')

      ds = tf.data.TFRecordDataset(tfrecords).map(parse_fn, num_parallel_calls=AUTO)
    else:
      tfrecord_filename = home/'ListingImageClassification'/'working_dir'/'tfrecords'/tfrecord_filename
      ds = tf.data.TFRecordDataset(tfrecord_filename).map(parse_fn, num_parallel_calls=AUTO)

    if return_decode_image:
      def decode_image(x):
        img = tf.image.decode_jpeg(x['image_raw'], channels=3)
        return img, x['filename'], x['orig_aspect_ratios']    # image in 1st slot to enable .predict(...) nicely.

      ds = ds.map(decode_image, num_parallel_calls=AUTO)
      return ds

    return ds

  def get_big_archive_soft_labels(self) -> pd.DataFrame:
    '''
    Return soft labels (aka predictions) for "big archive images" 

    Note: This should cover all but 192 images for get_big_archive_tfrecords_ds(), 
          the rest is covered by data_loader.get_image_labels_df()
    '''

    SamsungT5_deployment_predictions_df = pd.read_csv(home/'ListingImageClassification'/'working_dir'/'predictions'/'SamsungT5_deployment_predictions_df.csv', dtype={'listingId': str})
    deployment_predictions_df = pd.read_csv(home/'ListingImageClassification'/'working_dir'/'predictions'/'deployment_predictions_df.csv', dtype={'listingId': str})

    deployment_predictions_df = pd.concat([deployment_predictions_df, SamsungT5_deployment_predictions_df], axis=0, ignore_index=True)

    return deployment_predictions_df


  # mostly labelled dataset from all_hydra training
  def get_all_hydra_labels_ds(self, return_decode_image: bool = True) -> tf.data.Dataset:
    '''
    TFRecord tf.data.Dataset that was used to train the hydra model. 99% of them should have humna labels
    Nameing format and location
          home/'ListingImageClassification'/'data'/'all_for_hydra'/'all_labels_*.tfrecords'
    '''
    feature_desc = {
      'filename': TFRecordHelper.DataType.STRING,
      'image_raw': TFRecordHelper.DataType.STRING,
    }
    parse_fn = TFRecordHelper.parse_fn(feature_desc=feature_desc)

    tfrecords = (home/'ListingImageClassification'/'data'/'all_for_hydra').lf('all_labels_*.tfrecords')
    ds = tf.data.TFRecordDataset(tfrecords).map(parse_fn, num_parallel_calls=AUTO)

    if return_decode_image:
      def decode_image(x):
        img = tf.image.decode_jpeg(x['image_raw'], channels=3)
        return img, x['filename']    # image in 1st slot to enable .predict(...) nicely.

      ds = ds.map(decode_image, num_parallel_calls=AUTO)
      return ds
    
    # ds should have length 169,956
    # filenames = [filename.numpy().decode('utf-8') for img, filename in tqdm(ds)]
    return ds

  def get_all_hydra_labels(self) -> pd.DataFrame:
    '''
    Return all labels for all images in the hydra dataset
    '''
    image_labels_df = self.get_image_labels_df()
    return image_labels_df


  # images that are archived in 100x100 grid from image tagging pipeline
  def get_tagged_images_ds(self, tfrecord_filename: str = None, return_decode_image: bool = True) -> Dict[str, tf.data.Dataset]:
    '''
    TFRecord tf.data.Dataset from image tagging system. These are stored on external disk for now (My Mac Backup)
    Naming format and location
          /Volumes/My Mac Backup/RLP/ListingImageClassification/data/bigstack_rlp_listing_images_tfrecords/*_bigstack_rlp_listing_images_100.tfrecords

          Note: 673_bigstack_rlp_listing_images_100.tfrecords are from recent listings that have AVM quick quotes.
    '''
    
    if tfrecord_filename is None:
      bigstack_tfrecord_dir = Path('/Volumes/My Mac Backup/RLP/ListingImageClassification/data/bigstack_rlp_listing_images_tfrecords')
      assert bigstack_tfrecord_dir.exists(), f'{bigstack_tfrecord_dir} does not exist'
      # bigstack_tfrecord_photos_dir = bigstack_tfrecord_dir/'photos'
      tfrecords = bigstack_tfrecord_dir.lf('*.tfrecords')
    else:
      tfrecords = [tfrecord_filename]

    _features = {
      'filenames': TFRecordHelper.DataType.VAR_STRING_ARRAY,
      'image_raw': TFRecordHelper.DataType.STRING,
    }

    tfrecord_filename_2_ds = {}   # return this at the end

    for tfrecord in tfrecords:
      spec = TFRecordHelper.element_spec(tf.data.TFRecordDataset(tfrecord), return_keys_only=True)
      if 'orig_aspect_ratios' in spec:
        features = {**_features, **{'orig_aspect_ratios': TFRecordHelper.DataType.VAR_FLOAT_ARRAY}}
        def decode_image(x):
          return x['filenames'], tf.image.decode_jpeg(x['image_raw'], channels=3), x['orig_aspect_ratios']

        def unstack(filenames, bigImg, orig_aspect_ratios):    # unstack grid of NxN images
          batch_size = tf.shape(filenames)[0]

          img = tf.transpose(tf.reshape(tf.transpose(tf.reshape(bigImg, (SQRT_BATCH_SIZE, 416, 416*SQRT_BATCH_SIZE, 3)), (0, 2, 1, 3)), (-1, 416, 416, 3)), (0, 2, 1, 3))

          if batch_size != BATCH_SIZE:   # pad filenames, orig_aspect_ratios
            pad_size = BATCH_SIZE - batch_size
            out_filenames =  tf.concat([tf.sparse.to_dense(filenames), tf.zeros((pad_size,), dtype=tf.string)], axis=0)
            out_orig_aspect_ratios = tf.concat([tf.sparse.to_dense(orig_aspect_ratios), -1*tf.ones((pad_size,), dtype=tf.float32)], axis=0)
          else:
            out_filenames = tf.sparse.to_dense(filenames)
            out_orig_aspect_ratios = tf.sparse.to_dense(orig_aspect_ratios)

          return out_filenames, img, out_orig_aspect_ratios

        def rearrange_img_first(filename, img, r):
          return img, filename, r
      else:
        features = _features
        def decode_image(x):
          return x['filenames'], tf.image.decode_jpeg(x['image_raw'], channels=3)

        def unstack(filenames, bigImg):
          batch_size = tf.shape(filenames)[0]

          img = tf.transpose(tf.reshape(tf.transpose(tf.reshape(bigImg, (SQRT_BATCH_SIZE, 416, 416*SQRT_BATCH_SIZE, 3)), (0, 2, 1, 3)), (-1, 416, 416, 3)), (0, 2, 1, 3))

          if batch_size != BATCH_SIZE:
            pad_size = BATCH_SIZE - batch_size
            out_filenames =  tf.concat([tf.sparse.to_dense(filenames), tf.zeros((pad_size,), dtype=tf.string)], axis=0)
          else:
            out_filenames = tf.sparse.to_dense(filenames)

          return out_filenames, img

        def rearrange_img_first(filename, img):
          return img, filename

      parse_fn = TFRecordHelper.parse_fn(features)
      tile_img_ds = tf.data.TFRecordDataset(tfrecord).map(parse_fn).map(decode_image, num_parallel_calls=AUTO)
      img_ds = tile_img_ds.map(unstack).unbatch()

      # rearrange data tuple position
      img_ds = img_ds.map(rearrange_img_first, num_parallel_calls=AUTO)

      # batch_size = 32
      # batch_img_ds = img_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

      tfrecord_filename_2_ds[Path(tfrecord).name] = img_ds
      
    return tfrecord_filename_2_ds

  def get_tagged_images_soft_labels(self, bundle_id: int = None) -> Union[pd.DataFrame, List]:
    '''
    Get soft labels for tagged images

    If no "bundle_id" is specified, return a list of available pandas feather files
    If a "bundle_id" is specified, return a pandas dataframe of soft labels
    '''

    if bundle_id is None:
      return (home/'ListingImagesData'/'soft_labels').lf('*_bigstack_rlp_listing_images_100_predictions_df')
    else:
      return pd.read_feather(home/'ListingImagesData'/'soft_labels'/f'{bundle_id}_bigstack_rlp_listing_images_100_predictions_df')


 # These are low resolution google street view images (IROC project)
  def get_iroc_image_artifacts(self, img_height=375, img_width=375) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List, tf.data.Dataset, tf.data.Dataset]:
    def get_subfolder(img_name):
      return sha256digest(img_name)[:2]

    labels = ['detached', 'semi_detached', 'townhouse', 'condo', 'others']
    other_labels = ['trees', 'commercial', 'road', 'too_ambiguous']

    image_lineage_df = pd.read_feather(home/'ListingImageClassification'/'data'/'iroc'/'images'/'image_lineage_df')
    print(f'len(image_lineage_df): {image_lineage_df.shape[0]}')    

    iroc_home_links_sample_42_df = pd.read_feather(home/'ListingImageClassification'/'data'/'iroc'/'iroc_home_links_sample_42_df')
    print(f'len(iroc_home_links_sample_42_df): {iroc_home_links_sample_42_df.shape[0]}')

    iroc_property_df = pd.read_feather(home/'ListingImageClassification'/'data'/'iroc'/'iroc_property_df')
    print(f'len(iroc_property_df): {iroc_property_df.shape[0]}')

    human_labels_df = pd.read_csv(home/'ListingImageClassification'/'labels'/'all_iroc_labels_df.csv', dtype={'listingId': object})
    human_labels_df.annotation.fillna('', inplace=True)
    human_labels_df = human_labels_df.q_py("human_y.notnull()").copy()

    def merge_to_others(label):
      if label in other_labels:
        return 'others'

      return label

    human_labels_df.human_y = human_labels_df.human_y.apply(merge_to_others)
    print(f'len(human_labels_df): {human_labels_df.shape[0]}')

    feature_desc = {
      'filename': TFRecordHelper.DataType.STRING,
      'image_raw': TFRecordHelper.DataType.STRING
    }

    parse_fn = TFRecordHelper.parse_fn(feature_desc = feature_desc)  

    def decode_img(x):
      img = tf.image.decode_jpeg(x['image_raw'], channels=3)
      img = tf.cast(img, tf.float32)

      img.set_shape([img_height, img_width, 3])
      return img, x['filename']

    def decode_resize_img(x):
      img = tf.image.decode_jpeg(x['image_raw'], channels=3)
      img = tf.image.resize(img, [img_height, img_width], tf.image.ResizeMethod.BICUBIC, antialias=True)
      img = tf.cast(img, tf.float32)

      img.set_shape([img_height, img_width, 3])
      return img, x['filename']

    train_tfrecord_file = home/'ListingImageClassification'/'data'/'iroc'/'train.tfrecords'
    dev_tfrecord_file = home/'ListingImageClassification'/'data'/'iroc'/'dev.tfrecords'

    train_tfrecord_ds = tf.data.TFRecordDataset(train_tfrecord_file).map(parse_fn, num_parallel_calls=AUTO)
    train_img_ds = train_tfrecord_ds.map(decode_img, num_parallel_calls=AUTO)

    dev_tfrecord_ds = tf.data.TFRecordDataset(dev_tfrecord_file).map(parse_fn, num_parallel_calls=AUTO)
    dev_img_ds = dev_tfrecord_ds.map(decode_img, num_parallel_calls=AUTO)

    return image_lineage_df, iroc_home_links_sample_42_df, iroc_property_df, human_labels_df, labels, train_img_ds, dev_img_ds
 
  def get_iroc_images_ds(self, tfrecord_filename: str = None, img_height: int = 375, img_width: int = 375) -> tf.data.Dataset:
    '''
    TFRecord tf.data.Dataset from IROC image scraping.
    Original image location:
          home/'ListingImageClassification'/'data'/'iroc'/'images'

    Naming format and location
          home/'ListingImageClassification'/'data'/'iroc'/'images'/'iroc_images_000*_of_0017.tfrecords'

    '''
    features = {
      'filename': TFRecordHelper.DataType.STRING,
      'image_raw': TFRecordHelper.DataType.STRING,   # bytes for the encoded jpeg, png, etc.
    }

    parse_fn = TFRecordHelper.parse_fn(features)

    if tfrecord_filename is None:
      tfrecord_files = (home/'ListingImageClassification'/'data'/'iroc'/'images').lf('iroc_images_*_of_0017.tfrecords')
    else:
      tfrecord_files = [tfrecord_filename]

    ds = tf.data.TFRecordDataset(tfrecord_files).map(parse_fn, num_parallel_calls=AUTO)

    if img_height == 375 and img_width == 375:
      def decode_img(x):
        img = tf.image.decode_jpeg(x['image_raw'], channels=3)
        img = tf.cast(img, tf.float32)

        img.set_shape([img_height, img_width, 3])
        return img, x['filename']
    else:
      def decode_img(x):
        img = tf.image.decode_jpeg(x['image_raw'], channels=3)
        img = tf.image.resize(img, [img_height, img_width], tf.image.ResizeMethod.BICUBIC, antialias=True)
        img = tf.cast(img, tf.float32)

        img.set_shape([img_height, img_width, 3])
        return img, x['filename']

    ds = ds.map(decode_img, num_parallel_calls=AUTO)

    return ds

  def get_iroc_images_soft_labels(self) -> pd.DataFrame:
    '''
    Returns a dataframe with the soft labels for the IROC images.
    '''
    iroc_predictions_df = pd.read_feather(home/'ListingImageClassification'/'data'/'iroc'/'images'/'iroc_predictions_df')
    return iroc_predictions_df





  def __del__(self):
    self._close_db_connection()

  def _close_db_connection(self):
    print('Closing database connection')
    self.db_conn.close()
    self.engine.dispose()
