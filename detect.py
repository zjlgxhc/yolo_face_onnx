import copy
import os.path
import time
import cv2
import onnxruntime
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class YOLO(object):
    def __init__(self, onnx_filename):
        self.mean = None
        self.std = None
        self.n_classes = 1
        self.class_names = ['face']
        #使用GPU推理
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print('Using GPU:', onnxruntime.get_device())
        print('Using ONNXRuntime version:', onnxruntime.__version__)
        print('Using ONNX model:', onnx_filename)
        self.session = onnxruntime.InferenceSession(onnx_filename, providers=self.providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name
        self.imgsz = self.session.get_inputs()[0].shape[2:]

    def inference(self, image, nms_thr=0.5, conf_thr=0.5):
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(image)#8ms
        data = self.session.run([self.output_names], {self.input_name: input_image})[0]
        dets = self.multiclass_nms(data, origin_h, origin_w, nms_thr, conf_thr)
        if dets is not None:
            final_boxes = dets[:, :4] if len(dets) else np.array([])
            final_scores = dets[:, 4] if len(dets) else np.array([])
            final_cls_inds = dets[:, 5] if len(dets) else np.array([])
        return dets, image
    def preprocess_image(self, raw_bgr_image):
        image_raw = raw_bgr_image
        origin_h,origin_w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.imgsz[0] / origin_w
        r_h = self.imgsz[1] / origin_h
        r = min(r_h, r_w)
        if r_h > r_w:
            tw = self.imgsz[0]
            th = int(r_w * origin_h)
            tx1 = tx2 = 0
            ty1 = int((self.imgsz[1] - th) / 2)
            ty2 = self.imgsz[1] - th - ty1
        else:
            tw = int(r_h * origin_w)
            th = self.imgsz[1]
            tx1 = int((self.imgsz[0] - tw) / 2)
            tx2 = self.imgsz[0] - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        image=image.astype(np.float32)
        image_raw = image_raw.astype(np.float32)
        return image, image_raw, origin_h,origin_w


    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.imgsz[0] / origin_w
        r_h = self.imgsz[1] / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.imgsz[0] - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.imgsz[1] - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.imgsz[0] - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.imgsz[1] - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def multiclass_nms(self, prediction, origin_h, origin_w, iou_thres, conf_thres):
        boxes = prediction[prediction[:, :, 4] >= conf_thres]  # boxes包含置信度大于置信度阈值的框,69个框
        boxes[:, 0:4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4]) # 每个框的左上角和右下角坐标
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1) # 左上角X坐标
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1) # 右下角X坐标
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1) # 左上角y坐标
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1) # 右下角y坐标
        obj_conf = boxes[:, 4:5]  # 每个检测框的置信度，conf_scores
        cls_conf = boxes[:, 5:]  # 每一类的类别概率, cls_scores
        scores = obj_conf * cls_conf  # 总的置信分数, conf_scores*cls_scores
        sort_score = np.argsort(scores, axis=0)[::-1]
        boxes = boxes[sort_score]
        boxes = boxes[:, 0, :]
        class_indices = boxes[:, 5:].argmax(axis=1)
        confidences = boxes[:, 4]  # 目标置信度
        box_coords = boxes[:, :4]  # 目标框
        boxes = np.column_stack((box_coords, confidences, class_indices))  # 每个检测框有((x1,y1),(x2,y2),obj_conf,cls_is)组成
        keep_boxes = []  # 保留的boxes,即输出的检测框
        while boxes.shape[0]:
            iou = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > iou_thres  # 计算iou>iou阈值的,保留大于阈值的
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = iou & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        dets = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return dets



def rainbow(size=50):
    cmap = plt.get_cmap('jet')
    color_list = []
    for n in range(size):
        color = cmap(n / size)
        color_list.append(color[:3])
    return np.array(color_list)


def vis_frame(img, dets):
    boxes = dets[:, :4] if len(dets) else np.array([])
    scores = dets[:, 4] if len(dets) else np.array([])
    cls_ids = dets[:, 5] if len(dets) else np.array([])
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        _COLORS = rainbow(80).astype(np.float32).reshape(-1, 3)
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format('face', score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img

