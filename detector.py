import cv2
import time
import numpy as np

def detect(config):
  frame = config['frame']
  net = config['net']
  output_layers = config['output_layers']

  # get detections
  blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), crop=False)
  net.setInput(blob)
  
  height, width, _ = frame.shape

  start_time = time.time()
  outputs = net.forward(output_layers)
  end_time = time.time()
  print('[INFO] Detection Time: {}'.format(end_time - start_time))

  # get and filter boxes
  boxes = []
  confidences = []
  class_ids = []

  confidence_thresh = 0.5

  for output in outputs:
    for detection in output:
      
      scores = detection[5:]
      class_id = np.argmax(scores)
      if class_id != 0: # only keep person class names (person = 0)
        continue
      
      confidence = scores[class_id].item() # gets native python float from numpy val
      if confidence > confidence_thresh:
        centre_x = detection[0] * width
        centre_y = detection[1] * height
        box_width = int(detection[2] * width)
        box_height = int(detection[3] * height)
        left = int(centre_x - box_width / 2.0)
        top = int(centre_y - box_height / 2.0)
        box = [left, top, box_width, box_height]
        boxes.append(box)
        confidences.append(confidence)
        class_ids.append(class_id)

  iou_thresh = 0.3

  # use nms to remove overlapping boxes
  indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, iou_thresh)

  if indices == ():
    return [], [], []

  indices = indices.flatten()

  # store filtered boxes
  filt_boxes = []
  filt_confidences = []
  filt_class_ids = []

  for i in indices:
    filt_boxes.append(boxes[i])
    filt_confidences.append(confidences[i])
    filt_class_ids.append(class_ids[i])

  return filt_boxes, filt_confidences, filt_class_ids