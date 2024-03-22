# Creating database 
# It captures images and stores them in datasets  
import cv2, sys, numpy, os 
haar_file = 'model\haarcascade_frontalface_default.xml'
import cv2

# Create LBPH face recognizer
model = cv2.face.LBPHFaceRecognizer_create()

  
# All the faces data will be present in this folder 
datasets = 'images'  
  
sub_data = 'Charles'     
  
path = os.path.join(datasets, sub_data) 
if not os.path.isdir(path): 
    os.mkdir(path) 
  
# defining the size of images  
(width, height) = (130, 100)     
  
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0)  
  
# The program loops until it has 30 images of the face. 
count = 1
while count < 30:  
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height)) 
        cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
    count += 1
      
    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break