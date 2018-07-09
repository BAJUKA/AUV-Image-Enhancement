import cv2

import numpy as np

import matplotlib.pyplot as plt

from skimage import restoration
from PIL import Image







img = cv2.imread('24.jpg',-1)
print(img.shape)



rows,cols,dim=img.shape

(b,g,r) = cv2.split(img)

result = np.zeros((rows,cols,3))

result = cv2.merge([b,g,r])

img = result
cv2.imshow('image',img)
def homomorphic(img):

	img = np.float32(img)

	img = img/255

	rows,cols,dim=img.shape



	#rh,rl are high frequency and low frequency gain respectively.the cutoff 32 is kept for 512,512 images

	#but it seems to work fine otherwise

	rh, rl, cutoff = 1.8,0.5,32

	b,g,r = cv2.split(img)

	y_log_b = np.log(b+0.01)

	y_log_g = np.log(g+0.01)

	y_log_r = np.log(r+0.01)

	y_fft_b = np.fft.fft2(y_log_b)

	y_fft_g = np.fft.fft2(y_log_g)

	y_fft_r = np.fft.fft2(y_log_r)

	y_fft_shift_b = np.fft.fftshift(y_fft_b)

	y_fft_shift_g = np.fft.fftshift(y_fft_g)

	y_fft_shift_r = np.fft.fftshift(y_fft_r)



	#D0 is the cutoff frequency again a parameter to be chosen

	D0 = cols/cutoff

	H = np.ones((rows,cols))

	B = np.ones((rows,cols))

	for i in range(rows):

		for j in range(cols):

			H[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*D0**2))))+rl





	result_filter_b = H * y_fft_shift_b

	result_filter_g = H * y_fft_shift_g

	result_filter_r = H * y_fft_shift_r



	result_interm_b = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_b)))

	result_interm_g = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_g)))

	result_interm_r = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_r)))



	result_b = np.exp(result_interm_b)

	result_g = np.exp(result_interm_g)

	result_r = np.exp(result_interm_r)



	result = np.zeros((rows,cols,dim))

	result[:,:,0] = result_b

	result[:,:,1] = result_g

	result[:,:,2] = result_r

	ma=-1

	mi = 500

	for i in range(3):

		r = max(np.ravel(result[:,:,i]))

		x = min(np.ravel(result[:,:,i]))

		if r > ma:

			ma=r

		if x < mi:

			mi = x

	#print(mi)

	#print(result[0,0,:])

	result = (result)

	#print(result[0,0,:])

	#result = np.uint8(result)

	#norm_image = cv2.normalize(result,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

	return(result)

    	
def adapt_histogram(img):

	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(2,2))

	img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])



	result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	return(result)

######contrast sttttttttttttarttttttttttt
def normalizeRed(intensity):

    iI      = intensity

   

    minI    = 50

    maxI    = 230

 

    minO    = 0

    maxO    = 255

 

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return iO

 

# Method to process the green band of the image

def normalizeGreen(intensity):

    iI      = intensity

   

    minI    = 90

    maxI    = 225

 

    minO    = 0

    maxO    = 160

 

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return iO

 

# Method to process the blue band of the image

def normalizeBlue(intensity):

    iI      = intensity

   

    minI    = 50

    maxI    = 255

 

    minO    = 0

    maxO    = 150

 

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return iO

 

 

# Create an image object

#imageObject     = Image.open("24.jpg")
####################################error1 assign imageObject equal to result1
result1 = adapt_histogram(img)

imageObject=result1


# Split the red, green and blue bands from the Image

multiBands = imageObject.split()

 

# Apply point operations that does contrast stretching on each color band

normalizedRedBand      = multiBands[0].point(normalizeRed)

normalizedGreenBand    = multiBands[1].point(normalizeGreen)#######sequnce shoule be change if it is bgr? error 2

normalizedBlueBand     = multiBands[2].point(normalizeBlue)

 

# Create a new image from the contrast stretched red, green and blue brands

normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))

 

# Display the image before contrast stretching

#imageObject.show()

 

# Display the image after contrast stretching

#normalizedImage.show()
###################end contrastt

smooth = cv2.GaussianBlur(result2,(15,15),0)
res = homomorphic(smooth)
img=cv2.cvtColor(res,cv2.COLOR_BGR2YCR_CB)
################work on y component in ycbcr
#((pixel – min) / (max – min))*255.
img[:,:,0] = cv2.equalizeHist(img[:,:,0])
img[:,:,0]=cv2.medianBlur(img[:,:,0],5)
img[:,:,0]=cv2.Laplacian(img[:,:,0],cv2.CV_64F)
(y,cb,cr)=cv2.split(img)#y cb,cr are 2D matrix so problem is here #####error3
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
blue_image= merged.copy() # Make a copy
blue_image[:,:,0] = 0
blue_image[:,:,1] = 0

#cv2.imshow('img',)
#cv2.imwrite('edges.png',edges)
cv2.waitKey(0)

cv2.destroyAllWindows()#########remaining add later
