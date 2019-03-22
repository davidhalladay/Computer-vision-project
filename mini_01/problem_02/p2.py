import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV

#loading training data into np.array[6*40][56*46]
data = []

for i in range(1,41):
    for j in range(1,7):
        tmp_img = cv.imread('./p2_data/'+str(i)+'_'+str(j)+'.png',0)
        data.append(tmp_img.reshape(-1))

data = np.array(data)
print("number of trainning data: %s" %(str(data.shape)))
#######################################################
#                 (1)mean,eigenfaces
#######################################################
mean_face = np.mean(data, axis=0).reshape(1,56*46)

pca = PCA(n_components = 239)
eigenVectors = pca.fit(data - mean_face.reshape(1,56*46))

#save mean image
mean_img = mean_face.reshape(56,46)
plt.figure(figsize = (10,8))
plt.subplot(1,5,1)
plt.title("mean")
plt.imshow(mean_img,cmap = 'gray')

#save eigenface images
for i in range(0,4):
    eigen_img = np.reshape(eigenVectors.components_[i],newshape = (56,46))
    plt.subplot(1,5,i+2)
    plt.title("eigenface : %d" %(i+1))
    plt.imshow(eigen_img,cmap = 'gray')
plt.savefig('./p1_mean_eigenface.png')

#######################################################
#         (2)Plot the four reconstructed images
#######################################################
img_1_1 = cv.imread('./p2_data/1_1.png',0)
img_1_1 = img_1_1.reshape(1,56*46)

my_list = [3, 45, 140, 229]
plt.figure(figsize = (12,9))
k =1
for index in my_list:
    img = mean_img.reshape(1,56*46)
    for i in range(index):
        img += np.inner(img_1_1 - mean_img.reshape(1,56*46),eigenVectors.components_[i])*eigenVectors.components_[i]
    MSE = mean_squared_error(img_1_1,img)
    plt.subplot(1,4,k)
    plt.title("n = %d , MSE = %f" %(index,MSE))
    plt.imshow(img.reshape(56,46),cmap = 'gray')
    k += 1
plt.savefig('./p2_reconstruct.png')


#######################################################
#         (3)k-nearest neighbors algorithm
#######################################################
param_grid = {'n_neighbors' : [1,3,5]}
n_list = [3, 45, 140]
#training_data setting
train_data = eigenVectors.transform(data - mean_face)
train_label = np.array([ i for i in range(1,41) for j in range(6)])

gsearch = GridSearchCV(KNN() ,param_grid, cv=3)
for n in n_list:
    gsearch.fit( train_data[:,:n], train_label )
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.cv_results_['rank_test_score'])


#######################################################
#         (4)k-nearest neighbors algorithm
#######################################################
#get testing_data
test_data = []
for i in range(1,41):
    for j in range(7,11):
        tmp_img = cv.imread('./p2_data/'+str(i)+'_'+str(j)+'.png',0)
        test_data.append(tmp_img.reshape(-1))
test_data = np.array(test_data)
#testing_data setting
test_data = eigenVectors.transform(test_data - mean_face)
test_label = np.array([ i for i in range(1,41) for j in range(7,11)])

#testing
k = 1
n = 45
kNN = KNN(n_neighbors = k)
kNN.fit(train_data[:,:n],train_label)
y_pred = kNN.predict(test_data[:,:n])
#calculate accuracy
count = 0.
for i in range(y_pred.shape[0]):
    if(y_pred[i] == test_label[i]):
        count += 1
print("accuracy : %f " %(count/y_pred.shape[0]))
