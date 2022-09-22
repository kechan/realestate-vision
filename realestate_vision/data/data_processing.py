import subprocess, tarfile, re, os, shutil
from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union, Callable
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
import pandas as pd
from tfrecord_helper.tfrecord_helper import TFRecordHelper, TFRecordHelperWriter 

from realestate_core.common.utils import load_from_pickle, save_to_pickle
from ..common.utils import get_listingId_from_image_name
from .data_loader import DataLoader

from realestate_vision_nlp.data.data_processing import (gen_samples_for_downloading, 
                                                        get_jumpIds, 
                                                        return_clean_sample_images,
                                                        gen_predictions_for,
                                                        update_jumpIds_cache
                                                        )

from realestate_vision_nlp.common.utils import get_listing_subfolder_from_listing_id

from realestate_vision_nlp.data.data_processing import create_tfrecord_from_image_list as create_tfrecord_from_list

from realestate_vision_nlp.data.data_sampling import sample_listings_with_neg_remarks, sample_listings_with_price_less_than_qq
from realestate_vision_nlp.data.data_sampling import sample_listings_with_pos_remarks


from realestate_core.common.run_config import home, bOnColab

def create_tfrecord_from_image_list(imgs: Union[List[Path], str], 
                                    out_tfrecord_filepath: str = 'image.tfrecords', 
                                    height=416, 
                                    width=416,
                                    shard_size=10000) -> Callable:
  '''
  imgs: list of image paths (of type pathlib.Path or str) with assumptions that image actually exists at that path
  out_tfrecord_filepath: full path name of output tfrecord file
  height: height of image
  width: width of image
  shard_size: number of images in each shard

  returns: Callable that parse the tfrecord file

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

  return parse_fn



class CurateHighResAVMSnapshotListingImagesPipeline:
  '''
  This image curation pipeline was created in the context of Condition/Sentiment analysis and recognition of real estate images.
  It is such that full listing info and corresponding high res images are available.
  '''
  def __init__(self, image_dir: Path, data_dir: Path, avm_snapshot_listing_df: pd.DataFrame = None) -> None:
    '''
    image_dir: directory where images are stored
    tmp_dir: directory where temporary files can be stored
    avm_snapshot_listing_df: dataframe of avm_snapshot_listing_df from AVM monitoring
    '''
    self.image_dir = image_dir
    self.data_dir = data_dir
    self.avm_snapshot_listing_df = avm_snapshot_listing_df

    self.new_predictions_df = None

  def sample_and_download_from_jumplisting_com(self, n_sample: int = 500, sampling_method: str = 'RANDOM', qq_factor: float = 0.4) -> pd.DataFrame:
    '''
    n_sample: number of images to sample
    '''

    assert self.avm_snapshot_listing_df is not None, 'avm_snapshot_listing_df is not provided'

    if sampling_method == 'RANDOM':
      sample_listing_df = gen_samples_for_downloading(self.avm_snapshot_listing_df, self.image_dir, n_sample=n_sample)

    elif sampling_method == 'NEGATIVE_TERMS_IN_REMARK':
      sample_listing_df, _ = sample_listings_with_neg_remarks(self.avm_snapshot_listing_df)

      excl_jumpIds = get_jumpIds(self.image_dir)

      sample_listing_df.drop(index=sample_listing_df.q_py("jumpId.isin(@excl_jumpIds)").index, inplace=True)

      if sample_listing_df.shape[0] > n_sample:
        sample_listing_df = sample_listing_df.sample(n_sample, random_state=42).copy()

      sample_listing_df.defrag_index(inplace=True)

    elif sampling_method == 'PRICE_BELOW_QUICKQUOTE':
      sample_listing_df, _ = sample_listings_with_price_less_than_qq(self.avm_snapshot_listing_df, factor=qq_factor)

      excl_jumpIds = get_jumpIds(self.image_dir)

      sample_listing_df.drop(index=sample_listing_df.q_py("jumpId.isin(@excl_jumpIds)").index, inplace=True)

      if sample_listing_df.shape[0] > n_sample:
        sample_listing_df = sample_listing_df.sample(n_sample, random_state=42).copy()

      sample_listing_df.defrag_index(inplace=True)

    elif sampling_method == 'CT':   # carriage trade
      excl_jumpIds = get_jumpIds(self.image_dir)

      sample_listing_df = self.avm_snapshot_listing_df.q_py("carriageTrade and ~jumpId.isin(@excl_jumpIds)").copy() # sample(n=225, random_state=42).copy()
      
      if sample_listing_df.shape[0] > n_sample:
        sample_listing_df = sample_listing_df.sample(n_sample, random_state=42).copy()
      sample_listing_df.defrag_index(inplace=True)

    elif sampling_method == 'POSITIVE_TERMS_IN_REMARK':
      sample_listing_df, _ = sample_listings_with_pos_remarks(self.avm_snapshot_listing_df)

      excl_jumpIds = get_jumpIds(self.image_dir)

      sample_listing_df.drop(index=sample_listing_df.q_py("jumpId.isin(@excl_jumpIds)").index, inplace=True)

      if sample_listing_df.shape[0] > n_sample:
        sample_listing_df = sample_listing_df.sample(n_sample, random_state=42).copy()

      sample_listing_df.defrag_index(inplace=True)

    elif sampling_method == 'REMARK_WITH_POOR_ROBERTA_LARGE_SCORE':
      raise NotImplementedError
    else:
      raise NotImplementedError

    # print out list of manual tasks
    manual_task_note = """
    0) Save to tmp file to be uploaded to jumptools VM
       sample_listing_df.to_feather(tmp/'to_be_downloaded_from_jumptool_vm_df')

    1) Remove previous artifacts from jumptool VM
       rm -f avm_listing_images.tar.gz*
       rm to_be_downloaded_from_jumptool_vm_df

    2) Upload to_be_downloaded_from_jumptool_vm_df to jumptool VM
       scp to_be_downloaded_from_jumptool_vm_df kelvin@172.16.3.110:/home/kelvin/photos_tagging

    3) Run the script to download images to avm_listing_images.tar.gz and split
       nohup python tar_avm_listing_images.py > tar_avm_listing_images.log 2>&1 &
       nohup split -b 200m avm_listing_images.tar.gz avm_listing_images.tar.gz_ &

    4) Download to local machine.
       scp kelvin@172.16.3.110:/home/kelvin/photos_tagging/avm_listing_images.tar.gz_* .
       avoid doing this on a google drive dir.

    5) cat splits to restore avm_listing_images.tar.gz and upload to gs://ai-tests/tmp
       cat avm_listing_images.tar.gz_* > avm_listing_images.tar.gz
       gsutil cp avm_listing_images.tar.gz gs://ai-tests/tmp
    """
    
    print(manual_task_note)    

    return sample_listing_df

  def downlaod_images_tar_to_local(self) -> None:
    '''
    Download from gs://ai-tests/tmp/avm_listing_images.tar.gz to local
    '''
    gsutil_cmd = 'gsutil cp gs://ai-tests/tmp/avm_listing_images.tar.gz .'
    subprocess.call(gsutil_cmd, shell=True)

  def extract_images_tar(self, local_dir: Path = Path('/content/photos')) -> None:
    '''
    Extracts avm_listing_images.tar.gz to local_dir (default to /content/photos), should be done on colab
    '''

    assert bOnColab, "Must be run on Colab"

    with tarfile.open('avm_listing_images.tar.gz', 'r:gz') as f:
      f.extractall(str(local_dir))

  def gen_predictions(self, local_image_dir: Path = Path('/content/photos'), out_tfrecord_filepath: str = 'img_416x416.tfrecords') -> pd.DataFrame:
    '''
    Generate predictions for images in local_image_dir (default to /content/photos), should be run on colab
    '''
    assert bOnColab, "Must be run on Colab"

    imgs = return_clean_sample_images(image_dir=local_image_dir)
    create_tfrecord_from_list(imgs, out_tfrecord_filepath='img_416x416.tfrecords', height=416, width=416)
    try:
      new_predictions_df = gen_predictions_for('img_416x416.tfrecords')  
    except: # hack, try again upon an exception will work.
      new_predictions_df = gen_predictions_for('img_416x416.tfrecords')

    self.new_predictions_df = new_predictions_df
    self.imgs = imgs

    return new_predictions_df

  def merge_predictions(self):
    if self.new_predictions_df is None:
      raise ValueError("self.new_predictions_df is None, run gen_predictions first")

    orig_predictions_df = pd.read_feather(self.data_dir/'predictions_df')
    print(f'len(orig_predictions_df): {len(orig_predictions_df)}')

    predictions_df = pd.concat([orig_predictions_df, self.new_predictions_df], axis=0, ignore_index=True)

    predictions_df.to_feather(self.data_dir/'predictions_df')
    print(f'len(predictions_df): {len(predictions_df)}')

    print(f'New predictions merged. Resetting self.new_predictions_df and self.imgs to None')
    self.new_predictions_df = None
    
  def merge_jumpIds_w_high_res_images(self):
    # get high res image listings
    jumpIds_w_high_res_images = list(set([get_listingId_from_image_name(Path(f).name) for f in self.imgs]))

    # merge
    orig_jumpIds_w_high_res_images = load_from_pickle(self.data_dir/'jumpIds_w_high_res_images.pkl')
    print(f'len(orig_jumpIds_w_high_res_images): {len(orig_jumpIds_w_high_res_images)}')

    jumpIds_w_high_res_images = orig_jumpIds_w_high_res_images + jumpIds_w_high_res_images
    save_to_pickle(jumpIds_w_high_res_images, self.data_dir/'jumpIds_w_high_res_images.pkl')
    print(f'len(jumpIds_w_high_res_images): {len(jumpIds_w_high_res_images)}')

    print(f'New jumpIds_w_high_res_images merged')

  def save_predicted_images_gdrive(self):
    # store predicted images to gdrive
    with tarfile.open('avm_listing_images.tar.gz', 'w:gz') as f:
      for img_path in self.imgs:    
        img_name = Path(img_path).name
        tar_path = str(Path(img_path.replace('/content/photos/', '')).parent)

        f.add(img_path, arcname=f'{tar_path}/{img_name}')

    with tarfile.open('avm_listing_images.tar.gz', 'r:gz') as f:
      f.extractall(str(self.image_dir))

    update_jumpIds_cache(image_dir='/content/photos', cache_dir=self.image_dir)

    print(f'Resetting self.imgs to None')
    self.imgs = None


  def rebuild_avm_high_res_img_tfrecord(self) -> None:

    data_loader = DataLoader()
    avm_high_res_predictions_df = data_loader.get_avm_high_res_images_soft_labels()
    jumpIds_w_high_res_images = load_from_pickle(self.data_dir/'jumpIds_w_high_res_images.pkl')

    # new listings not in avm_high_res_predictions_df
    new_listings = list(set(jumpIds_w_high_res_images) - set(avm_high_res_predictions_df.listingId.values))

    missings = []
    imgs = []
    for listing_id in tqdm(new_listings):
      subfolder = get_listing_subfolder_from_listing_id(listing_id=listing_id, src_image_dir=self.image_dir)

      if subfolder is not None:
        imgs += subfolder.ls()
      else:
        missings.append(listing_id)

    create_tfrecord_from_image_list(imgs, 'avm_high_res_img.tfrecords')   # create at local

    # merge new tfrecords with existing set

    tfrecord_folder = home/'ConditionSentiment'/'data'/'images'/'tfrecords'

    d_all = int(re.compile('avm_high_res_img_\d+_of_(\d+).tfrecords').match(Path('.').lf('avm_high_res_img_*_of_*.tfrecords')[0].name).group(1))

    for f in tfrecord_folder.lf('avm_high_res_img_*_of_*.tfrecords'):
      fname = f.name
      match = re.compile('avm_high_res_img_(\d+)_of_(\d+).tfrecords').match(fname)
      if match is not None:
        ind = int(match.group(1))
        all = int(match.group(2))

        os.rename(f, tfrecord_folder/f'avm_high_res_img_{ind:04d}_of_{all+d_all:04d}.tfrecords')

    for f in Path('.').lf('avm_high_res_img_*_of_*.tfrecords'):
      fname = f.name
      match = re.compile('avm_high_res_img_(\d+)_of_\d+.tfrecords').match(fname)
      if match is not None:
        ind = int(match.group(1))

        shutil.copy(f, tfrecord_folder/f'avm_high_res_img_{ind+all:04d}_of_{all+d_all:04d}.tfrecords')

    soft_labels_prediction_file = home/'ConditionSentiment'/'data'/'images'/'soft_labels'/'avm_high_res_predictions_df'

    orig_predictions_df = pd.read_feather(soft_labels_prediction_file)

    for f in Path('.').lf('avm_high_res_img_*_of_*.tfrecords'):
      try:
        predictions_df = gen_predictions_for(f)  
      except: # hack, try again upon an exception will work.
        predictions_df = gen_predictions_for(f)

      orig_predictions_df = pd.concat([orig_predictions_df, predictions_df], axis=0, ignore_index=True)

    orig_predictions_df.to_feather(soft_labels_prediction_file)
      



class ImageTaggingAcquisitionPipeline:
  def __init__(self, image_dir: Path, tmp_dir: Path):
    '''
    Pipline to acquire new images from imaging tagging system.

    image_dir: directory where images are stored
    tmp_dir: directory where temporary files can be stored
    '''
    self.image_dir = image_dir
    self.tmp_dir = tmp_dir
