#!/usr/bin/env python
## Author: Rohit
## Date: July, 25, 2017
# Purpose: Ros node to detect objects using tensorflow
import matplotlib.pyplot as plt
import os
import sys
import cv2
import numpy as np
import argparse
from PIL import Image
from collections import defaultdict
import h5py
import time

try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    print("  sudo apt install python-pip")
    print("  sudo pip install tensorflow")
    sys.exit(1)

# ROS related imports
#import rospy
#from std_msgs.msg import String , Header
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError
#from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

print("--- All imports successful ---")

# SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.4

######### Set model here ############
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
#MODEL_NAME =  'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME =  'exported_model2'
# By default models are stored in data/models/
#MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]), MODEL_NAME)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(MODEL_PATH, 'frozen_inference_graph.pb')
######### Set the label map file here ###########
LABEL_NAME = 'bottle_label_map.pbtxt'
# By default label maps are stored in data/labels/
#PATH_TO_LABELS = os.path.join(os.path.dirname(sys.path[0]), 'data', 'labels', LABEL_NAME)
PATH_TO_LABELS = os.path.join(BASE_DIR, 'data/labels', LABEL_NAME)
######### Set the number of classes here #########
NUM_CLASSES = 3
######### Set the image directory here #########
IMG_PATH = os.path.join(BASE_DIR, 'images')
CROP_COLOR_PATH = os.path.join(BASE_DIR, 'simulation/color')
CROP_DEPTH_PATH = os.path.join(BASE_DIR, 'simulation/depth')
CROP_AFFORD_PATH = os.path.join(BASE_DIR, 'simulation/afford')
CROP_RESULT_PATH = os.path.join(BASE_DIR, 'simulation/result')
######### Set the lua model directory here #########
PATH_TO_LUA = os.path.join(BASE_DIR, 'affordance_model', 'infer.lua')


# crops_dict should be a defaultdict(dict)
def detectCrop(crops_dict):
    start_all = time.time()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print('-------======-------')
    ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`,
    # we know that this corresponds to `airplane`.  Here we use internal utility functions,
    # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Setting the GPU options to use fraction of gpu that has been set
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
        )


    # Detection
    sess = tf.Session(graph=detection_graph,config=config)
    imgDir = IMG_PATH

    imgNames = [f for f in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, f)) and f[0] != '.']
    for imgName in imgNames:
        if args.img in imgName:
            print("--- Processing img {} ---".format(imgName))

            imgIdx = imgName[:4]
            imgPath = os.path.join(imgDir, imgName)
            depPath = os.path.join(imgDir, '{}_depth.png'.format(imgIdx))
            image = cv2.imread(imgPath)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = np.asarray(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            #print('begin detect ...')
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            #print('end detect ....')

            boxes_to_label, image_show =vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                #min_score_thresh=[0.4, 0.3, 0.3],
                use_normalized_coordinates=True,
                line_thickness=2)

            # # show the original color image with boxes outlines
            # plt.imshow(cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB))
            # plt.show()

            boxes = [i for i in boxes_to_label if boxes_to_label[i]!=boxes_to_label.default_factory()]
            print("--- img_{} {} boxes detected ---".format(imgIdx, len(boxes)))

            # get size of original image
            img_height, img_width, _ = image.shape
            crops_dict[imgIdx]['width'] = img_width
            crops_dict[imgIdx]['height'] = img_height
            # initialize an empty list for this image to contain the name of all its crops
            crops_dict[imgIdx]['crops'] = []
            crops_dict[imgIdx]['result'] = os.path.join(CROP_RESULT_PATH, '{}_result.jpg'.format(imgIdx))

            for i in range(len(boxes)):
                # each boxes[i] is a tuple of coordinates
                if len(boxes[i]) != 0:
                    ymin, xmin, ymax, xmax = boxes[i]
                    (left, right, top, bottom) = (xmin * img_width, xmax * img_width,
                                                   ymin * img_height, ymax * img_height)
                    (left, right, top, bottom) = (int(round(left)), int(round(right)), int(round(top)), int(round(bottom)))

                    # open color and depth images
                    fp = open(imgPath, 'rb')
                    fp_depth = open(depPath, 'rb')
                    image_o = Image.open(fp)
                    image_d = Image.open(fp_depth)

                    # crop and save the color and depth crops
                    cropIdx = '{}_{}'.format(imgIdx, i)
                    img_c = image_o.crop([left, top, right, bottom])
                    img_c_d = image_d.crop([left, top, right, bottom])
                    crop_color_path = os.path.join(CROP_COLOR_PATH, '{}_color.jpg'.format(cropIdx))
                    crop_depth_path = os.path.join(CROP_DEPTH_PATH, '{}_depth.png'.format(cropIdx))
                    img_c.save(crop_color_path)
                    img_c_d.save(crop_depth_path)

                    # add attributes to crops_dict
                    # add the name of the crop to the original image imgIdx
                    crops_dict[imgIdx]['crops'].append(cropIdx)
                    # add attributes to each crop cropIdx
                    crops_dict[cropIdx]['color'] = crop_color_path
                    crops_dict[cropIdx]['depth'] = crop_depth_path
                    crops_dict[cropIdx]['afford'] = os.path.join(CROP_AFFORD_PATH, '{}.h5'.format(cropIdx))
                    crops_dict[cropIdx]['outcrop'] = os.path.join(CROP_AFFORD_PATH, '{}.jpg'.format(cropIdx))
                    crops_dict[cropIdx]['left'] = left
                    crops_dict[cropIdx]['right'] = right
                    crops_dict[cropIdx]['top'] = top
                    crops_dict[cropIdx]['bottom'] = bottom

            # process affordance for the entire image_o
            time_th = runTorch(crops_dict, imgIdx)
            print("--- img_{} affordance finished processing ---".format(imgIdx))

            # print out some time stats
            end_all = time.time()
            time_all = round(end_all - start_all - time_th, 2) 
            print("--- Torch took {}s on average ---".format(time_th))
            print("--- Rest of prog took {}s ---".format(time_all))
            

# given an imgIdx (e.g. 0001) runTorch returns its affordance map
def runTorch(crops_dict, imgIdx):
    if imgIdx not in crops_dict.keys():
        raise Exception('[!!!] invalid imgIdx')

    # init an empty red jpg here, same size as image_o
    size_o = (crops_dict[imgIdx]['width'], crops_dict[imgIdx]['height'])
    outimg = Image.new('RGB', size_o, (255, 0, 0))
    
    # for each of the crops from this imgIdx, first make h5, 
    # then plot h5 as jpg, then overlay it on top of original jpg
    cropIndices = crops_dict[imgIdx]['crops']
    time_th = []
    for cropIdx in cropIndices:

        # get the crop image resolution
        crop_width = crops_dict[cropIdx]['right'] - crops_dict[cropIdx]['left']
        crop_height = crops_dict[cropIdx]['bottom'] - crops_dict[cropIdx]['top']
        assert (crop_width > 0 and crop_height > 0)

        start_th = time.time()
        
        # run Torch commands, from infer.lua located at PATH_TO_LUA
        command = 'th ' + PATH_TO_LUA + ' -imgColorPath ' + crops_dict[cropIdx]['color'] +\
        ' -imgDepthPath ' + crops_dict[cropIdx]['depth'] + ' -resultPath ' + crops_dict[cropIdx]['afford'] +\
        ' -imgHeight ' + str(crop_height) + ' -imgWidth ' + str(crop_width)
        
        # print(command)
        os.system(command)
        
        # Check if Torch created .h5 is created
        if not os.path.isfile(crops_dict[cropIdx]['afford']):
            raise Exception('[!!!] h5 result not created')
        
        end_th = time.time()
        time_th.append(end_th - start_th)

        # convert h5 result to jpg
        size = (crop_width, crop_height)
        outjpg = crops_dict[cropIdx]['outcrop']
        crop_afford = h5toJpg(crops_dict[cropIdx]['afford'], outjpg, size)

        # overwrite outimg with affordance map
        offset = (crops_dict[cropIdx]['left'], crops_dict[cropIdx]['top'])
        outimg.paste(crop_afford, offset)

    # save the affordance map of whole image_o
    outimg.save(crops_dict[imgIdx]['result'], "JPEG")

    # Calculate average time Torch took per crop
    time_th_ave = reduce(lambda x: x+y, time_th)/len(time_th) 
    time_th_ave = round(time_th_ave, 2)
    return time_th_ave


def h5toJpg(inh5, outjpg, size):
    f = h5py.File(inh5, 'r')

    # List all group keys
    a_group_key = list(f.keys())[0]

    # plot jpg from h5 file
    dset = f[a_group_key]
    data = np.array(dset[:,:,:])
    # PIL takes in (w,h,c) uint8 0-255
    data = np.transpose(data[0], (1,2,0))
    data = data*255
    img = Image.fromarray(data.astype(np.uint8))
    # resize outcrop to original size
    img = img.resize(size)
    img.save(outjpg, "JPEG")

    # check if affordance outcrop created
    if not os.path.isfile(outjpg):
        raise Exception('[!!!] outjpg not created')

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', default='_color.jpg', help='specify color img name to crop, e.g. 0001_color.jpg (default returns every jpg')
    args = parser.parse_args()
    
    # init empty dict of dict
    crops_dict = defaultdict(dict)
    # main prog
    detectCrop(crops_dict)
