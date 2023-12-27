import time
import cv2 
from flask import Flask, render_template, Response,request

from utils import label_map_util
from utils import visualization_utils as vis_util
from flask import *
import numpy as np
import tensorflow as tf
import os


app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()

MODEL_NAME = 'inference_graph'


CWD_PATH = os.getcwd()


PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to image


# Number of classes the object detector can identify
NUM_CLASSES = 3


@app.route('/',methods=['POST','GET'])
def index():
    """Video streaming home page."""
    return render_template('video_analysis.html')






def Object_detection(image_file):
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

      sess = tf.Session(graph=detection_graph)
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  image = image_file
  image_expanded = np.expand_dims(image, axis=0)
  (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_expanded})
  gun_count=0
  tank_count=0
  soldier_count=0
  vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
  return image


def gen():
    count = 0
    cap = cv2.VideoCapture('1.mp4')
    while(cap.isOpened()):
        count = 5
        cap.set(1, count)
        ret, frame = cap.read()
        if ret:
            image=Object_detection(frame)
            cv2.imshow('Detection',image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        else
            print('File not found ')
        '''frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
           break'''
