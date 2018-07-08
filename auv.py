import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration

img = cv2.imread('images/5.png',-1)
img = cv2.resize(img,(512,512))




print(img.shape)

rows,cols,dim=img.shape
b,g,r,alpha = cv2.split(img)
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
	result[:,:,2] = r
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

def contrast(im):
	m_b = max(np.ravel(im[:,:,0])) 
	mi_b = min(np.ravel(im[:,:,0]))
	for i in range(im.shape[0]):
		for j in range(img.shape[1]):
			im[i,j,0] = (im[i,j,0]-mi_b)*255/(m_b-mi_b)

	m_g = max(np.ravel(im[:,:,1])) 
	mi_g = min(np.ravel(im[:,:,1]))
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			im[i,j,1] = (im[i,j,1]-mi_g)*255/(m_g-mi_g)

	m_r = max(np.ravel(im[:,:,2])) 
	mi_r = min(np.ravel(im[:,:,2]))
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			im[i,j,2] = (im[i,j,2]-mi_r)*255/(m_r-mi_r)
			
	return(im)
#####this method of contrast is good I decrease  blue<green<red show minimize the blue effect and we can easily change itt
####READ IT CONTRAST22222222
##########contrast222222222 starttttttttttttttttt
# Method to process the red band of the image

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

imageObject=img 

# Split the red, green and blue bands from the Image

multiBands      = imageObject.split()
# Apply point operations that does contrast stretching on each color band

normalizedRedBand      = multiBands[0].point(normalizeRed)

normalizedGreenBand    = multiBands[1].point(normalizeGreen)

normalizedBlueBand     = multiBands[2].point(normalizeBlue)

# Create a new image from the contrast stretched red, green and blue brands

normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))

# Display the image before contrast stretching

imageObject.show()

# Display the image after contrast stretching

normalizedImage.show()


###########################contrast 222222222222222222222 endddddddd
result1 = adapt_histogram(img)
cv2.imshow('result1',result1)
######################its better to paste here .imageObject=result2
result2 = contrast(result1)####instead of this
cv2.imshow('result2',result2)
smooth = cv2.GaussianBlur(result2,(15,15),0)
cv2.imshow('smooth',smooth)
res = homomorphic(smooth) 
cv2.imshow('result',res)

cv2.waitKey(0)
cv2.destroyAllWindows()
#upper code not working with jpg that 24.jpg image
#below code is working butby using cv2.imwrite(res24.png,res)
#not working but running sepearetly pllz link both code  #doooo dude  
#this code is not working wit
pimg=cv2.imread('res24.jpg',1)


def from_pil(pimg):
    #pimg = pimg.convert(mode='RGB')
    nimg = np.asarray(pimg)
    nimg.flags.writeable = True
    return(nimg)


def stretch_pre(nimg):
    """
    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Colour Slides: An Empirical Evaluation'
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0]-nimg[0].min(),0)
    nimg[1] = np.maximum(nimg[1]-nimg[1].min(),0)
    nimg[2] = np.maximum(nimg[2]-nimg[2].min(),0)
    return nimg.transpose(1, 2, 0)

def gray_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)




nimg=from_pil(pimg)
nimg=stretch_pre(nimg)
final=gray_world(nimg)
cv2.imwrite('rgbclahegrey24.png',final)



























