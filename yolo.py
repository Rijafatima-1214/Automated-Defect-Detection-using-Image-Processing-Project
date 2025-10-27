import os
import numpy as np

def yolo_available():
    base = os.path.dirname(__file__)
    cfg = os.path.join(base, 'models','yolov4-tiny.cfg')
    weights = os.path.join(base, 'models','yolov4-tiny.weights')
    names = os.path.join(base, 'models','coco.names')
    return os.path.exists(cfg) and os.path.exists(weights) and os.path.exists(names)

def detect_yolo(image_path, conf_threshold=0.4, nms_threshold=0.4):
    import cv2
    base = os.path.dirname(__file__)
    cfg = os.path.join(base, 'models','yolov4-tiny.cfg')
    weights = os.path.join(base, 'models','yolov4-tiny.weights')
    names = os.path.join(base, 'models','coco.names')
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    img = cv2.imread(image_path)
    h,w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    ln = net.getLayerNames()
    out_names = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(out_names)
    boxes=[]
    class_ids=[]
    confidences=[]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([w,h,w,h])
                cx,cy,bw,bh = box.astype('int')
                x = int(cx - bw/2)
                y = int(cy - bh/2)
                boxes.append([x,y,int(bw),int(bh)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results=[]
    with open(names) as f:
        CLASS_NAMES = [l.strip() for l in f.readlines()]
    if len(idxs) > 0:
        try:
            it = idxs.flatten()
        except Exception:
            it = idxs
        for i in it:
            x,y,wc,hc = boxes[i]
            results.append({'bbox':[int(x),int(y),int(wc),int(hc)], 'type': CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else 'object', 'score': confidences[i]})
    return results
