from functools import partialmethod
import inspect, os
import PIL, base64
from io import BytesIO
import pandas as pd
import tensorflow as tf

def get_thumbnail(path):
  if os.name == 'nt':
    path = "\\\\?\\"+path # This "\\\\?\\" is used to prevent problems with long Windows paths

  i = PIL.Image.open(path)    
  return i

def image_base64(im):
  if isinstance(im, str):
    im = get_thumbnail(im)
  with BytesIO() as buffer:
    im.save(buffer, 'jpeg')
    return base64.b64encode(buffer.getvalue()).decode()

# def image_formatter(im):
#   return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'   
def image_formatter(im, width=100):
  return f'<img src="data:image/jpeg;base64,{image_base64(im)}" style="width: {width}px;">'



# pd.DataFramne.to_html_with_image = partialmethod(pd.DataFrame.to_html, formatters={'image': image_formatter}, escape=False)
# pd.DataFrame.to_html_with_image = lambda self, colname='image': self.to_html(formatters={colname: image_formatter}, escape=False)

pd.DataFrame.to_html_with_image = lambda self, colname='image', img_width=100: self.to_html(
    formatters={colname: lambda im: image_formatter(im, width=img_width)}, 
    escape=False
)

# def inspect_dataset_item(ds):
#   for item in ds.take(1):
#     if isinstance(item, tuple):
#       for i, x in enumerate(item):
#         if tf.is_tensor(x):
#           if x.ndim == 3 and (x.dtype == tf.uint8 or x.dtype == tf.float32):
#             print(f'{i}: image')
#           elif x.ndim == 0 and x.dtype == tf.string:
#             print(f'{i}: string')
#         else:
#           print(f'{i}: {x}, not a tf.tensor')


# tf.data.Dataset.element_spec = lambda self: inspect_dataset_item(self)
