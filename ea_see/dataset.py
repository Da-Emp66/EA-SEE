import cv2
import dlib

from torch.utils.data import Dataset


class FaceRecognitionDataset(Dataset):
    def __init__(self):
        pass

    def fabricate(self):
        prebuilt_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        img = cv2.imread('path_to_image')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = prebuilt_face_detector(gray, 1)
        left, top, right, bottom = 0, 0, 0, 0

        for rect in rects:
            left = rect.rect.left()
            top = rect.rect.top()
            right = rect.rect.right()
            bottom = rect.rect.bottom()
            width = right - left
            height = bottom - top
            
            img_crop = img[top:top+height, left:left+width]
            cv2.imwrite('path_to_image_as_person_name', img_crop)
