#!/usr/bin/env python
## Author: Rohit
## Date: July, 25, 2017
# Purpose: Ros node to detect objects using tensorflow
import matplotlib.pyplot as plt
import os
import sys
import cv2
import numpy as np

import PIL.Image



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

# SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.4

######### Set model here ############
#MODEL_NAME =  'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME =  'exported_model2'
# By default models are stored in data/models/
#MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]), MODEL_NAME)
MODEL_PATH = "/home/thu/subSda100/xxx/trash_detection/tensorflow_object_detector/" + MODEL_NAME
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
######### Set the label map file here ###########
LABEL_NAME = 'bottle_label_map.pbtxt'
# By default label maps are stored in data/labels/
#PATH_TO_LABELS = os.path.join(os.path.dirname(sys.path[0]), 'data', 'labels', LABEL_NAME)
PATH_TO_LABELS = "/home/thu/subSda100/xxx/trash_detection/tensorflow_object_detector/" + 'data/labels/' + LABEL_NAME
######### Set the number of classes here #########
NUM_CLASSES = 3


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
imgDir = "/home/thu/subSda100/xxx/trash_detection/tensorflow_object_detector/images/"

for imgName in os.listdir(imgDir):
    print(imgName)
    if 'jpg' in imgName:
        imgPath = imgDir + imgName
        #depPath = imgDir + imgName[:5] + 'depth.png'
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
        # print(len(scores[0]))
        # print(boxes[0])
        print(imgName)
        print(len(classes[0]))

        boxes_to_label, image_show =vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            #min_score_thresh=[0.4, 0.3, 0.3],
            use_normalized_coordinates=True,
            line_thickness=2)
        
        plt.imshow(cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB))
        plt.show()
        print(classes)
        boxes = [i for i in boxes_to_label if boxes_to_label[i]!=boxes_to_label.default_factory()]
        print(boxes_to_label)

        im_height, im_width, _ = image.shape

        print(im_height, im_width)
        #print(im_width, im_height)
        #print(boxes)
        # if use_normalized_coordinates:

        for box in boxes:
            #box = tuple(box.tolist())
            if len(box) != 0:
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                               ymin * im_height, ymax * im_height)
                print(left, right, top, bottom)
                fp = open(imgPath, 'rb')
                #fp_depth = open(depPath, 'rb')
                image_o = PIL.Image.open(fp)
                #image_d = PIL.Image.open(fp_depth)
                plt.imshow(image_o)
                plt.plot([left, right, left, right], [top, top, bottom, bottom], 'ro')
                plt.show()
                img_c = image_o.crop([left, top, right, bottom])
                #img_c_d = image_d.crop([left, top, right, bottom])
                crop_color_path = imgDir + imgName[:5] + 'color_crop.jpg'
                #crop_depth_path = imgDir + imgName[:5] + 'depth_crop.png'
                #img_c.save(crop_color_path)
                #img_c_d.save(crop_depth_path)
                plt.axis('off')
                plt.imshow(img_c)
                plt.show()






