# White Walker in Python!
# Author: Amin Aghaee
# Homepage: http://cs.ubc.ca/~aghaee/
# Summer 2017

import numpy as np
import cv2

# Specify name of your file here:
filename = 'face.jpg'
# -------------------------------
pallete_threshold = 0.99


def correlated(color, pallete, threshold):
    for k in range( np.size(pallete,0) ):
        COS = (np.dot(color, pallete[k])) / (np.linalg.norm(color) * np.linalg.norm(pallete[k]));
        if COS >= threshold:
            return True
    return False        

    
# Turn skin desaturated:
def desat(image, percent):
    pallete = np.array([[255,224,189], [255,205,148], [234,192,134], [255,173,96], [255,227,159]])

    desat = image
    sz = image.shape
    
    grey = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
    
    for i in range(sz[0]):
        for j in range(sz[1]):
            if correlated(image[i,j], pallete, pallete_threshold) == False:
                desat[i,j] = percent * grey[i,j] + (1-percent)*image[i,j]
            else:
                 desat[i,j] = image[i,j]   
    return desat


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread(filename)
r = 500.0 / img.shape[1]  
dim = (500, int(img.shape[0] * r))
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


resized = desat(resized, 0.8)
grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(grey, 1.3, 5)

for (x,y,w,h) in faces:      
    roi_grey = grey[y:y+(3*h/5), x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_grey)
for (ex,ey,ew,eh) in eyes:  
    for i in range( np.floor(ex).astype(int), ex+ew):
        for j in range(np.floor(ey+0.3*eh).astype(int),ey+eh):
            if grey[y+j,x+i] > 220:
                resized[y+j,x+i] = 0.7 * np.array([252, 226, 157]) + 0.3*resized[y+j,x+i]
            else: 
                if grey[y+j,x+i] < 60:
                    resized[y+j,x+i] = 0.6 * np.array([239, 145, 37]) + 0.4*resized[y+j,x+i]


snow = cv2.imread('snow.png', 0)
dim = (500, int(resized.shape[0]))
snow = cv2.resize(snow, dim, interpolation = cv2.INTER_AREA)

for i in range(resized.shape[0]):
    for j in range(resized.shape[1]):
        if snow[i,j] > 0:
            resized[i,j] = 0.7 * resized[i,j] + 0.3 * snow[i,j]

cv2.imshow('img',resized)  
cv2.imwrite('output.jpg',resized)
cv2.waitKey(0) 

cv2.destroyAllWindows()