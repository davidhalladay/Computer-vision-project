import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# (2) 2D Gaussian filter using a 3 × 3 kernel
img = cv.imread('./lena.png',0)

Gaussian_img = cv.GaussianBlur(img,(3,3),sigmaX = 1/(2*np.log(2)),sigmaY = 0)

plt.figure(figsize = (10,8))
plt.subplot(121),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.subplot(122),plt.imshow(Gaussian_img,cmap = 'gray'),plt.title('Gaussian_img')
plt.savefig("./p4_Gaussian.png")

# (3)
kernel_x = np.array([1/2., 0. ,-1/2.])
kernel_y = np.array([1/2., 0. ,-1/2.]).T

# Original_img Im
Ix_img = cv.filter2D(img,-1,kernel_x)
Iy_img = cv.filter2D(img,-1,kernel_y)
Im_img = (Ix_img**2 + Iy_img**2 )**0.5

plt.figure(figsize = (10,8))
plt.subplot(221),plt.imshow(Ix_img,cmap = 'gray'),plt.title('Ix_img')
plt.subplot(222),plt.imshow(Iy_img,cmap = 'gray'),plt.title('Iy_img')
plt.subplot(223),plt.imshow(Im_img,cmap = 'gray'),plt.title('Im_img')

# Gaussian_img Im
Ix_img_G = cv.filter2D(Gaussian_img,-1,kernel_x)
Iy_img_G = cv.filter2D(Gaussian_img,-1,kernel_y)
Im_img_G = (Ix_img_G**2 + Iy_img_G**2 )**0.5
plt.subplot(224),plt.imshow(Im_img_G,cmap = 'gray'),plt.title('Im_img_G')
plt.savefig("./p4_Im.png")
