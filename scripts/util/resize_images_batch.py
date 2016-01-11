import os
from PIL import Image

src_path = "../../data/CUB/"
basewidth = 75

image_type = (".jpg", ".png", ".JPG", ".PNG", ".JPEG", ".tif", ".tiff", ".TIFF")

image_paths = []  
for root, dirs, files in os.walk(src_path):
  image_paths.extend([os.path.join(root, f) for f in files if f.endswith(image_type)])


def resize_img(img_path):
    img = Image.open(img_path)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img_name = img_path.replace("CUB", "CUB-rescaled")
    out_path = "/".join(img_name.split("/")[:-1])
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    img.save(out_path + "/" + img_name.split("/")[-1])

for image_path in image_paths:
    resize_img(image_path)