
import os
from PIL import Image

######### Set model here ############
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
IMG_PATH = os.path.join(BASE_DIR, 'scenario_images')

imgNames = [f for f in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, f)) and f[0] != '.']
for imgName in imgNames:
    print(imgName)
    imgPath = os.path.join(IMG_PATH, imgName)
    saveImg = os.path.join(BASE_DIR, 'resize', imgName)
    im = Image.open(imgPath)
    im = im.crop((250,250,1150,950))
    im.save(saveImg)

