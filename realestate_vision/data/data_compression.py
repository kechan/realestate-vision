''' Image compression using VQGAN''' 

import math
from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union, Callable
from pathlib import Path
from tqdm import tqdm

from realestate_core.common.run_config import home, bOnColab

import tensorflow as tf
import numpy as np

import yaml, io
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ

import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont

chkpt_log_dir = home/'ListingImagesData'/'vqgan'

import torch
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# declare utility functions
def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess(img: Image, target_image_size=256) -> torch.Tensor:
  ''' resize and center crop a PIL image and return a torch tensor '''
  s = min(img.size)
  
  if s < target_image_size:
    raise ValueError(f'min dim for image {s} < {target_image_size}')
      
  r = target_image_size / s
  s = (round(r * img.size[1]), round(r * img.size[0]))
  img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
  img = TF.center_crop(img, output_size=2 * [target_image_size])
  img = torch.unsqueeze(T.ToTensor()(img), 0)

  return img

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def vqgan_decoded_to_img_array(x: torch.Tensor, return_pil: bool = False):
  '''
  Convert a VQGAN decoded image to a numpy array
  '''
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  if x.ndim == 3:
    x = x.permute(1,2,0).numpy()
  else:
    x = x.permute(0, 2, 3, 1).numpy()

  x = (255*x).astype(np.uint8)

  if return_pil:
    x = Image.fromarray(x)
    if not x.mode == "RGB":
      x = x.convert("RGB")
  return x  

def reconstruct_with_vqgan(x: torch.Tensor, model: VQModel):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

def load_model1024() -> VQModel:
  config1024 = load_config(chkpt_log_dir/"vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
  model1024 = load_vqgan(config1024, ckpt_path=chkpt_log_dir/"vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)

  return model1024

def load_model16384() -> VQModel:
  config16384 = load_config(chkpt_log_dir/"vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
  model16384 = load_vqgan(config16384, ckpt_path=chkpt_log_dir/"vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)

  return model16384

def load_model32x32() -> VQModel:
  config32x32 = load_config(chkpt_log_dir/"vqgan_gumbel_f8/configs/model.yaml", display=False)
  model32x32 = load_vqgan(config32x32, ckpt_path=chkpt_log_dir/"vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to(DEVICE)

  return model32x32


def compress_image(img: tf.Tensor, model: VQModel) -> np.ndarray:
  '''
  Compress a single image represented by normalized tf.Tensor using VQModel and return the 
  latent indices. The image can be reconstructed with decoder given indices + codebook
  '''

  x = torch.from_numpy(img.numpy())
  x = x.unsqueeze(0).permute(0, 3, 1, 2)
  x = x.to(DEVICE)

  x = preprocess_vqgan(x)

  z, _, [_, _, indices] = model.encode(x)
  indices = indices.detach().cpu().numpy()

  return indices

def reconstruct_image_from_indices(indices: np.ndarray, model: VQModel) -> np.ndarray:
  '''
  Reconstruct a single image from indices and codebook
  '''
  indices = torch.from_numpy(indices).to(DEVICE)
  n = int(math.sqrt(indices.shape[0]))

  codebook = model.quantize.embedding.weight
  z = codebook[indices].reshape(n, n, codebook.shape[-1]).unsqueeze(0).permute(0, 3, 1, 2)

  recon_img = vqgan_decoded_to_img_array(model.decode(z)[0])

  return recon_img


def compress_images(imgs: tf.Tensor, model: VQModel) -> np.ndarray:
  '''
  Compress a batch of images represented by normalized tf.Tensor using VQModel and return the 
  latent indices. The images can be reconstructed with decoder given indices + codebook
  '''

  batch_size = imgs.shape[0]

  x_batch = torch.from_numpy(imgs.numpy())
  x_batch = x_batch.permute(0, 3, 1, 2)
  x_batch = x_batch.to(DEVICE)

  x_batch = preprocess_vqgan(x_batch)

  _, _, [_, _, indices] = model.encode(x_batch)

  indices = indices.reshape(batch_size, -1)

  indices = indices.detach().cpu().numpy()

  return indices


def reconstruct_images_from_indices(indices: np.ndarray, model: VQModel, batch_size: int = 16) -> np.ndarray:
  '''
  Reconstruct a batch of images from indices and codebook
  '''
  # print(torch.is_grad_enabled())
  codebook = model.quantize.embedding.weight
  n = int(math.sqrt(indices.shape[1]))
  n_batches = int(math.ceil(indices.shape[0] / batch_size))   # break indices into batches

  # indices = torch.from_numpy(indices).to(DEVICE)
  
  imgs = []
  for i in tqdm(range(n_batches)):
    batch_indices = indices[i*batch_size:(i+1)*batch_size]
    batch_indices = torch.from_numpy(batch_indices).to(DEVICE)

    z = codebook[batch_indices].reshape(-1, n, n, codebook.shape[-1]).permute(0, 3, 1, 2)
    # imgs.append(model.decode(z))
    imgs.append(vqgan_decoded_to_img_array(model.decode(z)))

  imgs = np.concatenate(imgs, axis=0)

  return imgs


def compress_image_tf_dataset(ds: tf.data.Dataset, model: VQModel, batch_size: int = None) -> np.ndarray:
  ''''
  Compress a tf dataset of images using VQGAN. 
  The dataset record should be a tuple of image shape (None, None, 3) and filename tensors
  '''
  if batch_size is not None:   # ds is not yet batched.
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  z_indices = []
  for img_batch, *_ in tqdm(ds):
    img_batch = tf.cast(img_batch, tf.float32) / 255.
    z_indices_batch = compress_images(img_batch, model)
    z_indices.append(z_indices_batch)

  z_indices = np.concatenate(z_indices, axis=0)

  return z_indices


