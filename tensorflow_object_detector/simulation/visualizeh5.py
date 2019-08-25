import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from PIL import Image


def runTorch(crops_dict, lua_path):
    for key in crops_dict.keys():
        height = str(crops_dict[key]['height'])
        width = str(crops_dict[key]['width'])
        
        command = 'th ' + lua_path + ' -imgColorPath ' + crops_dict[key]['color'] +\
        ' -imgDepthPath ' + crops_dict[key]['depth'] + ' -resultPath ' + crops_dict[key]['result'] +\
        ' -outputScale ' + '1' + ' -imgHeight ' + height + ' -imgWidth ' + width
        
        print(command)
        os.system(command)
        
        if isfile(crops_dict[key]['result']):
            # convert h5 result to jpg
            size = (crops_dict[key]['width'], crops_dict[key]['height'])
            convertJpg(crops_dict[key]['result'], crops_dict[key]['outimg'], size)
            
        else:
            # Use this to catch the exception for torch itself
            raise Exception('[!] h5 result not created')
            

def convertJpg(inh5, outjpg, size):
    f = h5py.File(inh5, 'r')

    # List all group keys
    a_group_key = list(f.keys())[0]

    # plot jpg from h5 file
    try:
        dset = f[a_group_key]
        data = np.array(dset[:,:,:])
        dataT = np.transpose(data[0], (1,2,0))
        plt.imsave(outjpg, dataT)
    except IOError:
        print ("Error creating '%s'" % outjpg)
    
    # resize output image
    try:
        img = Image.open(outjpg)
        new_img = img.resize(size)
        new_img.save(outjpg, "JPEG")
    except IOError:
        print ("Error resizing '%s'" % outjpg)
    
    
def getCropsDict(crop_color_dir, crop_depth_dir, crop_result_dir):
    # get all file names in crop_dir
    crops = [f[:4] for f in listdir(crop_color_dir) if isfile(join(crop_color_dir, f)) and f[0]!='.']
    
    crops_dict = defaultdict(dict)
    for f in crops:
        corlor_add = ''.join([f, '_color_crop.jpg'])
        crops_dict[f]['color'] = join(crop_color_dir, corlor_add)
        depth_add = ''.join([f, '_depth_crop.png'])
        crops_dict[f]['depth'] = join(crop_depth_dir, depth_add)
        result_add = ''.join([f, '_result.h5'])
        crops_dict[f]['result'] = join(crop_result_dir, result_add)
        outimg_add = ''.join([f, '_result.jpg'])
        crops_dict[f]['outimg'] = join(crop_result_dir, outimg_add)
        
        # get the image resolution
        with Image.open(crops_dict[f]['color']) as img:
            width, height = img.size
            crops_dict[f]['width'] = width
            crops_dict[f]['height'] = height
        
        # ensure color and depth images have the same resolutions
        with Image.open(crops_dict[f]['depth']) as img:
            width, height = img.size
            assert crops_dict[f]['width'] == width
            assert crops_dict[f]['height'] == height
        
    return crops_dict
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--color', help='corlor dir')
    parser.add_argument('--depth', help='depth dir')
    parser.add_argument('--result', help='result dir')
    parser.add_argument('--lua', help='dir path of infer.lua')
    args = parser.parse_args()
    
    crops_dict = getCropsDict(args.color, args.depth, args.result)
    runTorch(crops_dict, args.lua)

#crops_dict = getCropsDict("/Users/Naifu/Downloads/crop/crop_color", "/Users/Naifu/Downloads/crop/crop_depth", "/Users/Naifu/Downloads/crop/crop_depth")
#runTorch(crops_dict, '../affordance_model/infer.lua')