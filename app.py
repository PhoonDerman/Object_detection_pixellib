import pixellib
from pixellib.instance import instance_segmentation
segment_image = instance_segmentation()
segment_image.load_model('mask_rcnn_coco.h5')
segment_image.segmentImage("images.jfif", show_bboxes=True, output_image_name="output2.jpg")
