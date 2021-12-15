############ IMPORT PACKAGES
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
import platform
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import io

from scripts import storage_handlers,reports
####### LOAD TF MODEL


import tensorflow as tf
model_tf = tf.keras.models.load_model("mask_detector\mask_detector.model")
#calculate mask accuracy
labels_preds = ["mask", "No  mask"]
def predict_mask_acc(frame, bbox):
    #cut head
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
    y2 = int((y2-y1)/3)+y1
    face = frame[y1:y2, x1:x2]
    #calculate bounding box
    face = cv2.resize(face, (224, 224))
    face = np.asarray(face)
    face = tf.keras.applications.mobilenet.preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    #making predictions from face
    prediction_arr = model_tf.predict(face)[0]
    label_idx = np.argmax(prediction_arr)
    if(prediction_arr[label_idx]*100<60):
        prediction = "unk"
    else:
        prediction = labels_preds[np.argmax(prediction_arr)]+" "+str(round(prediction_arr[label_idx]*100,2))
    #TODO: SEND DATA TO SERVER
    return prediction


######### LOAD YOLO MODEL
#device = '0' # 'cpu'for CPU USAGE
device = 'cpu'
half = True
yolo_weights = 'yolov5n.pt'
dnn = False
imgsz = [640]
imgsz *= 2 if len(imgsz) == 1 else 1
# Initialize
device = select_device(device)
half &= device.type != 'cpu'  # half precision only supported on CUDA
# Load model
device = select_device(device)
model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size

############ DETECT FUNCTION

def detect(path):
    webcam = False
    # REPLACEMENTS:
    out = 'inference/output'
    #source = 'test.jpg'
    source = path
    yolo_weights = 'yolov5l.pt'
    show_vid = False  # only avaliable for webcams or video processing
    save_vid = True
    imgsz = [640]
    half = True
    hide_labels=False
    hide_conf=False,
    
    dnn = False
    visualize = False
    conf_thres = 0.4
    iou_thres = 0.5
    classes = 0  # FOR ONLY PERSONS
    agnostic_nms = True
    max_det = 1000
    augment = False
    
    imgsz *= 2 if len(imgsz) == 1 else 1
    
    ret_response = []
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    save_path = str(Path(out))
    
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for i_l,(*xyxy, conf, cls) in enumerate(reversed(det)):
                    if round(conf.item(),2)>0.6:  # check confidence image
                        c = int(cls)  # integer class
                        label2 = None if hide_labels else (names[c] if hide_conf else 'id: ')
                        label = 'id: '
                        conf_round = round(conf.item(),2)
                        #get predictions by id
                        prediction = predict_mask_acc(im0,xyxy)
                        #print(prediction)
                        temp_dict_ret  = dict()
                        temp_dict_ret['id_person'] = i_l

                            #cut head
                        x1, y1, x2, y2 = xyxy
                        x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                        temp_dict_ret['box_xy'] = (x1, y1, x2, y2)
                        temp_dict_ret['yolo_acc'] = conf_round
                        temp_dict_ret['mask_acc'] = prediction

                        ret_response.append(temp_dict_ret)
                        annotator.box_label(xyxy, label + str(i_l) +" "+ prediction, color=colors(c, True))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_vid:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    files_cloud = storage_handlers.upload_photos(reports.get_id())
                    print("file saved")
                
    return (ret_response,files_cloud)


#ASD
if __name__ == "__main__":
    with torch.no_grad():
        detect()



def get_predictions_img(img_byte):
    path = 'test.jpg'
    #preprocess
    image = Image.open(io.BytesIO(img_byte))
    image.save(path)
    #detect with yolov5 and mask model
    answer,files_cloud = detect(path)
    return (answer,files_cloud)


    
