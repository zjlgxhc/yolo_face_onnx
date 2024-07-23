import cv2
from detect import YOLO
from detect import vis_frame

#继承自YOLO
class yolo(YOLO):
    def __init__(self,onnx_filepath):
        super(yolo, self).__init__(onnx_filepath)
        self.class_names = ['face']
        self.n_classes = 1

def main():
    yolov5 = yolo('yolov5s.onnx')
    image = cv2.imread('001_fe3347c0.jpg')
    dets, image = yolov5.inference(image)
    vis_frame(image, dets)
    cv2.namedWindow('test')
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()