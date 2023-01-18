import cv2
import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance.segmentation()
segment_image.load_models('mask_rcnn_coco.h5')
camera=cv2.VideoCapture(0)

while camera.isOpened():
    res,frame=camera.read()
    result=segment_image.segmentFrame(frame,show_bboxes=True)
    image=result[1]
    cv2.imshow('Image Segmentation', image)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()