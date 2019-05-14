# object_detector.py

import cv2
import os
import numpy as np
import tensorflow as tf
import tarfile
import six.moves.urllib as urllib
import time
import xml.etree.ElementTree as ET
import sys
import argparse

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from lxml import etree

class ObjectDetector():
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
    NUM_CLASSES = 90

    def __init__(self, model_name, label_file='data/mscoco_label_map.pbtxt'):
        # Initialize some variables
        self.process_this_frame = True

        # download model
        self.graph_file = model_name + '/' + self.GRAPH_FILE_NAME
        if not os.path.isfile(self.graph_file):
            self.download_model(model_name)

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            graph = self.detection_graph

            ops = graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = graph.get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, 480, 640)
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            self.tensor_dict = tensor_dict

        self.sess = tf.Session(graph=self.detection_graph)

        # Loading label map
        # Label maps map indices to category names,
        # so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`.
        # Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(label_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.output_dict = None

        self.last_inference_time = 0

    def run_inference(self, image_np):
        sess = self.sess
        graph = self.detection_graph
        with graph.as_default():
            image_tensor = graph.get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(self.tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def time_to_run_inference(self):
        unixtime = int(time.time())
        if self.last_inference_time != unixtime:
            self.last_inference_time = unixtime
            return True
        return False

    def detect_objects(self, frame, output_path,filename):
        time1 = time.time()
        # Grab a single frame of video

        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        time2 = time.time()

        # Only process every other frame of video to save time
        if self.time_to_run_inference():
            self.output_dict = self.run_inference(rgb_small_frame)

        time3 = time.time()

        vis_util.visualize_boxes_and_labels_on_image_array(
          frame,
          self.output_dict['detection_boxes'],
          self.output_dict['detection_classes'],
          self.output_dict['detection_scores'],
          self.category_index,
          instance_masks=self.output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=1)
        height, width = frame.shape[:2]
        root=etree.Element("data")
        for i in range(0,20) :
            if (self.output_dict['detection_scores'][i]>0.25):
                ob=etree.Element("object")
                cid=etree.SubElement(ob,"id")
                cid.text=str(self.output_dict['detection_classes'][i])
                cs=etree.SubElement(ob,"score")
                cs.text=str(self.output_dict['detection_scores'][i])
                box=etree.SubElement(ob, "box")
                ymn=etree.SubElement(box,"ymin")
                ymn.text=str(int(self.output_dict['detection_boxes'][i][0]*height))
                xmn=etree.SubElement(box,"xmin")
                xmn.text=str(int(self.output_dict['detection_boxes'][i][1]*width))
                ymx=etree.SubElement(box,"ymax")
                ymx.text=str(int(self.output_dict['detection_boxes'][i][2]*height))
                xmx=etree.SubElement(box,"xmax")
                xmx.text=str(int(self.output_dict['detection_boxes'][i][3]*width))
                img_crop=frame[int(ymn.text):int(ymx.text),int(xmn.text):int(xmx.text)]
                blu = cv2.medianBlur(img_crop,5)
                gray=cv2.cvtColor(blu, cv2.COLOR_BGR2GRAY)
                wid=blu.shape[1]
                hei=blu.shape[0]
                max_val=0
                for i in range(10, wid-10, 5):
                    for j in range(10, hei-10, 5):
                        if gray[j,i]>max_val :
                            max_val=gray[j][i]
                            if max_val >250 :
                                break
                ret,th = cv2.threshold(gray,max_val-10,255,cv2.THRESH_BINARY)
                mask=np.array(th)
                masked_img=cv2.bitwise_and(blu,blu,mask=mask)
                counter=0
                sum_b=0
                sum_g=0;
                sum_r=0;
                hsv=cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
                sum_val=0
                for i in range(3,wid-3):
                    for j in range(5, hei-5, 3):
                        if hsv[j, i, 2]!= 0 & hsv[j,i,2]!=255:
                            sum_b=sum_b+masked_img[j,i,0]
                            sum_g=sum_g+masked_img[j,i,1]
                            sum_r=sum_r+masked_img[j,i,2]
                            sum_val= sum_val+(hsv[j,i,2])
                            # (j=y, i=x) 순서주의  add Hue value
                            counter = counter + 1
                rc=sum_r/counter
                gc=sum_g/counter
                bc=sum_b/counter
                val = int(sum_val/counter)
                c=etree.SubElement(ob,"color")
                b=etree.SubElement(c,"b")
                b.text=str(int(bc))
                g=etree.SubElement(c,"g")
                g.text=str(int(gc))
                r=etree.SubElement(c,"r")
                r.text=str(int(rc))
                v=etree.SubElement(ob,"intensity")
                strv=str(val/255)
                v.text=strv[0:7]
                root.append(ob)
            else :
                break
        
        x_output = etree.tostring(root, pretty_print=True, encoding='UTF-8')
        x_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
        f=open(output_path+'/'+filename+'.xml','w',encoding="utf-8") 
        f.write(x_header+x_output.decode('utf-8'))

        time4 = time.time()
        #print(time2-time1, time3-time2, time4-time3)
        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 로그 감추기
    ap = argparse.ArgumentParser()
    d=os.getcwd()
    ap.add_argument('image_file', help='image_file_to_run_inference')
    ap.add_argument('output_path', nargs='?', default=d, help='detected frame is saved to file')
    args = ap.parse_args()
    detector = ObjectDetector('ssd_inception_v2_coco', label_file='data/light_label_map.pbtxt')#모델경로 및 라벨맵경로수정 앞에가 모델이 들어있는 디렉토리
    t=True    
    img = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    filename=os.path.splitext(args.image_file)[-2]
    frame = detector.detect_objects(img,args.output_path,filename)
    print(args.output_path+"/"+filename+".xml") #이부분 변경가능하면
    cv2.imwrite(args.output_path+'/'+filename+'_output.jpg', frame)


