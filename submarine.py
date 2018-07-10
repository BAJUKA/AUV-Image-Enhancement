result2 = normalizedImage



smooth = result2 

res = homomorphic(smooth)
###########cannnnnnnnnnyyyyyyyy
res1 = cv2.GaussianBlur(res,(5,5),0)
cv2.imshow('finalimg',res)
#cv2.imwrite('finalimg.jpg',res)

#blueg=cv2.bilateralFilter(blue_image,50,50,50)
#blueg=cv2.GaussianBlur(res,(5,5),0)
canny=cv2.Canny(smooth,100,200)
###########looooooooooogggggggggggggg
blue_image= res.copy() # Make a copy
blue_image[:,:,1] = 0
blue_image[:,:,2] = 0
cv2.imwrite('blue_image.jpg',blue_image)
res3 = cv2.GaussianBlur(blue_image,(5,5),0)
log=cv2.Laplacian(res3,cv2.CV_64F)
cv2.imshow('cannyimg2',canny)
cv2.imshow('logimg2',log)
##pixel correction and wernier and image scaling remian doing it
