import cv2
import imutils
import numpy as np
import random
import colorsys

# inspiration and some code pieces were copied from https://github.com/haroonshakeel/yolo_get_preds/blob/master/my_utils.py

def get_random_bright_colors(size):
    for i in range(0,size-1):
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
        yield (r,g,b)


def get_yolo_preds(net, video_url, confidence_threshold, overlapping_threshold, labels = None, frame_resize_width=None):

    # List of colors to represent each class label with distinct bright color
    colors = list(get_random_bright_colors(len(labels)))

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    cap = cv2.VideoCapture(video_url)

    try:
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        yolo_width_height = (416, 416)

        counter = 0
        max_count = 0

        while True:
            (_, frame) = cap.read()

            counter += 1 
            
            if frame_resize_width:
                frame = imutils.resize(frame, width=frame_resize_width)
            (H, W) = frame.shape[:2]

            # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, yolo_width_height, swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > confidence_threshold:
                        # Scale the bboxes back to the original image size
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # Remove overlapping bounding boxes
            bboxes = cv2.dnn.NMSBoxes(
                boxes, confidences, confidence_threshold, overlapping_threshold)
            if len(bboxes) > 0:
                for i in bboxes.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in colors[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                    
                    # draw bounding box title background
                    text_offset_x = x
                    text_offset_y = y
                    text_color = (255, 255, 255)
                    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=1)[0]
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 80, text_offset_y - text_height + 4))
                    cv2.rectangle(frame, box_coords[0], box_coords[1], color, cv2.FILLED)
                    
                    # draw bounding box title
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            #if len(bboxes) > max_count:
            #    max_count = len(bboxes)
            #    cv2.imwrite('captured_' + str(counter) + '.jpg', frame) 
            cv2.imshow("YOLOv4 Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break the loop
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


with open("model/coco.names","r", encoding="utf-8") as f:
    labels = f.read().strip().split("\n")

yolo_config_path = "model/yolov4.cfg"
yolo_weights_path = "model/yolov4.weights"

useCuda = True

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

if useCuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

video_url = "https://cdn-004.whatsupcams.com/hls/hr_pula01.m3u8"
frame_width = 1200

if __name__ == '__main__':
    get_yolo_preds(net, video_url, 0.6, 0.1, labels,frame_width)