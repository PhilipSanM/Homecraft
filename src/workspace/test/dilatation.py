import cv2 as cv
import numpy as np



src = "../YOLOv/test/yolo8_00001.png"



element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * 3 + 1, 2 * 3 + 1),
                                    (3, 3))


dilatation =  cv.dilate(cv.imread(src), element, iterations=3)


cv.imwrite("../YOLOv/test/dilatation_yolo8_00001.png", dilatation)