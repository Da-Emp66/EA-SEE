import cv2
import dlib

# Load cnn_face_detector with 'mmod_face_detector'
prebuilt_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Load image 
img = cv2.imread('path_to_image')

# Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  		
# Find faces in image
rects = prebuilt_face_detector(gray, 1)
left, top, right, bottom = 0, 0, 0, 0

# For each face 'rect' provides face location in image as pixel loaction
for (i, rect) in enumerate(rects):
    left = rect.rect.left()
    top = rect.rect.top()
    right = rect.rect.right()
    bottom = rect.rect.bottom()
    width = right - left
    height = bottom - top

    # Crop image 
    img_crop = img[top:top+height, left:left+width]

    #save crop image with person name as image name 
    cv2.imwrite('path_to_image_as_person_name', img_crop)
