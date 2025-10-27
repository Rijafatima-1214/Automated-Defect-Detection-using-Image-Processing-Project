import cv2, os, numpy as np
from yolo import yolo_available, detect_yolo

def allowed_file(filename):
    ALLOWED = {'png','jpg','jpeg','bmp','tif','tiff'}
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED

def preprocess_image(img, width=1000):
    h, w = img.shape[:2]
    if w > width:
        scale = width / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    # equalize
    equalized = cv2.equalizeHist(denoised)
    return img, equalized

def classical_defect_detection(img_gray):
    # edges + morphology + contour filtering
    edges = cv2.Canny(img_gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 150: continue
        x,y,w,h = cv2.boundingRect(c)
        boxes.append({'bbox':[int(x),int(y),int(w),int(h)], 'type':'crack', 'score':None})
    return boxes

def draw_boxes(img, boxes):
    out = img.copy()
    for b in boxes:
        x,y,w,h = b['bbox']
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 2)
        label = b.get('type','defect')
        cv2.putText(out, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
    return out

def detect_defects(image_path, use_yolo_if_available=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Could not read image: {image_path}')
    img_resized, gray = preprocess_image(img)
    boxes = []
    if use_yolo_if_available and yolo_available():
        try:
            boxes = detect_yolo(image_path)
        except Exception:
            boxes = classical_defect_detection(gray)
    else:
        boxes = classical_defect_detection(gray)
    out = draw_boxes(img_resized, boxes)
    # save processed
    processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, 'proc_' + os.path.basename(image_path))
    cv2.imwrite(processed_path, out)
    return {'input_path': image_path, 'processed_path': processed_path, 'defects': boxes}
