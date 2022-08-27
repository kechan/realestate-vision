from functools import partialmethod
import PIL, base64
from io import BytesIO
import pandas as pd

def get_thumbnail(path):
  path = "\\\\?\\"+path # This "\\\\?\\" is used to prevent problems with long Windows paths
  i = PIL.Image.open(path)    
  return i

def image_base64(im):
  if isinstance(im, str):
    im = get_thumbnail(im)
  with BytesIO() as buffer:
    im.save(buffer, 'jpeg')
    return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
  return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'   


# pd.DataFramne.to_html_with_image = partialmethod(pd.DataFrame.to_html, formatters={'image': image_formatter}, escape=False)
pd.DataFrame.to_html_with_image = lambda self, colname='image': self.to_html(formatters={colname: image_formatter}, escape=False)