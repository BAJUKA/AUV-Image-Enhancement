import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import color
import bisect
from PIL import Image
#######first we need to correct it and add after till what we done 
img=cv2.imread('24.jpg',1)
#img =skimage.color.rgba2rgb(img)
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
#((pixel – min) / (max – min))*255.
img[:,:,0] = cv2.equalizeHist(img[:,:,0])
img[:,:,0]=cv2.medianBlur(img[:,:,0],5)
img[:,:,0]=cv2.Laplacian(img[:,:,0],cv2.CV_64F)
(y,cb,cr)=cv2.split(img)
################problem starts here y,cb,cr ar 2D matrix and wantedt to do i sent you image to convert ycbcr to rgb
y=np.matrix(y)
cb=np.matrix(cb)
cr=np.matrix(cr)
X=np.matrix([y,cb,cr])
Y=np.matrix([[16],[128],[128]])
P=np.matrix([[0.0046,0.0000,0.0063],[0.0046,-0.0015,0.0032],[0.0046,0.0079,0.0000]])
rgb=np.matrix([[ ],[ ],[ ]])
rgb=np.matmul(P,X-Y)
r=rgb[0,0]
b=rgb[1,0]
g=rgb[2,0]
merged=Image.merge("RGB",(r,g,b))
merged.show()
#cv2.imshow('img',rgb)
#cv2.waitKey(0)

#cv2.destroyAllWindows()
###########
