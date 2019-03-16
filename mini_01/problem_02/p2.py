import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2 as cv

#loading training data into np.array[6*40][56*46]
data = np.zeros(56*46)
for i in range(1,41):
    for j in range(1,7):
        tmp_img = cv.imread('./p2_data/'+str(i)+'_'+str(j)+'.png',0)
        data = np.vstack((data,tmp_img.reshape(1,56*46)))
data = np.delete(data,0,0)

#print(np.vstack((img_1.reshape(1,56*46),img_2.reshape(1,56*46))))


mean, eigenVectors = \
    cv.PCACompute(data, mean=None, maxComponents = 6*40)


mean_img = mean.reshape(56,46)
plt.imshow(mean_img,cmap = 'gray')
plt.show()

"""
print(img.shape)
plt.imshow(img,cmap = 'gray')
plt.show()

cv.namedWindow('image',cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destoryAllwindows()
"""
