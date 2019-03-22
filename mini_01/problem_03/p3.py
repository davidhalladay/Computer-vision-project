import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import random

#################################################
##                 loading data
#################################################
# load training_data
print("Start to loading training data...")
data_path = ["banana","fountain","reef","tractor"]
train_data_num = [str(i) for i in range(1,376)]
test_data_num = [str(i) for i in range(375,501)]
X_train = []
for i in range(len(data_path)):
    for j in range(375):
        tmp_img = cv.imread('./p3_data/'+data_path[i]+
        '/'+data_path[i]+'_'+train_data_num[j].zfill(3)+'.JPEG',1)
        X_train.append(tmp_img)
X_train = np.array(X_train)
print("number of trainning data: %s" %(str(X_train.shape)))
# load test data
print("Start to loading testing data...")
X_test = []
for i in range(len(data_path)):
    for j in range(125):
        tmp_img = cv.imread('./p3_data/'+data_path[i]+
        '/'+data_path[i]+'_'+test_data_num[j].zfill(3)+'.JPEG',1)
        X_test.append(tmp_img)
X_test = np.array(X_test)
print("number of trainning data: %s" %(str(X_test.shape)))

#################################################
##           into a grid of (16,16,3)
#################################################
x_rand = [0,1,2,3]
y_rand = [0,1,2,3]
data_rand = [i for i in range(375)]
category = [0,375,750,1125]
plt.figure(figsize = (10,8))
for i in range(4):
    keep = category[i]
    data_choose = random.choice(data_rand)
    for j in range(0,10,4):
        dx = random.choice(x_rand)
        dy = random.choice(y_rand)
        img_cut = X_train[keep + data_choose,dx*16:16*(dx+1),16*dy:16*(dy+1),::-1]
        plt.subplot(3,4,i+1+j)
        plt.title("data_choose : %d" %(data_choose+1))
        plt.imshow(img_cut)
plt.savefig('./p3_grid4.png')

#####
#loading train patches
X_train_patches = []
for n in X_train:
    for i in range(4):
        for j in range(4):
            tmp = n[16*i:16*(i+1),16*j:16*(j+1),:]
            X_train_patches.append(tmp.reshape(768,1))
X_train_patches = np.array(X_train_patches).reshape(24000,768)
print("number of trainning patches: %s" %str(X_train_patches.shape))
#loading test patches
X_test_patches = []
for n in X_test:
    for i in range(4):
        for j in range(4):
            tmp = n[16*i:16*(i+1),16*j:16*(j+1),:]
            X_test_patches.append(tmp.reshape(768,1))
X_test_patches = np.array(X_test_patches).reshape(8000,768)
print("number of testing patches: %s" %str(X_test_patches.shape))


#################################################
##           k-means algorithm
#################################################

print("-"*20)
print("Training the cluster model by k-means alg...")

k_means = KMeans(n_clusters=15 ,max_iter=5000)
k_means.fit(X_train_patches)

k_means_center = k_means.cluster_centers_
k_means_labels = k_means.labels_

# 3 - PCA
pca = PCA(n_components = 3)
pca_X_train_patches = pca.fit_transform(X_train_patches)


# random choice the cluster 6 from 15 clusters
sample_num = np.random.choice(range(15)
    , size = 6, replace = False)

# construst sampled center
sample_center = []
for i in sample_num: sample_center.append(k_means_center[i])
sample_center = np.array(sample_center)
sample_pca_center = pca.transform(sample_center)

# construst sampled patches
sample_patches = []
for i in sample_num:
    tmp = []
    for j in range(pca_X_train_patches.shape[0]):
        if (k_means_labels[j] ==  i):
            tmp.append(pca_X_train_patches[j])
    tmp = np.array(tmp)
    sample_patches.append(tmp)
sample_patches = np.array(sample_patches)

# plot 3D
print("setting data and plot data...")
fig = plt.figure(figsize=(16,8))
ax = Axes3D(fig)
sample_colors = ['b', 'g', 'r', 'c', 'm', 'y']
for i in range(6):
    tmp = sample_patches[i]
    for j in range(sample_patches[i].shape[0]):
        tmp_1 = tmp[j]
        ax.scatter(tmp_1[0], tmp_1[1], tmp_1[2]
            ,s = 10,c = sample_colors[i],alpha = 0.2)
    tmp_center = sample_pca_center[i]
    ax.scatter(tmp_center[0], tmp_center[1]
        , tmp_center[2],s=200,c = sample_colors[i],marker = 'D',alpha = 1.0)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
plt.savefig('./Axes3D.png')

#################################################
##                soft-Max pooling
#################################################
print("Starting setting X by Bow features...")
#setting features & training data
bow_features = k_means_center  #(15 * 768)
X_train_patches = X_train_patches  #(24000 * 768)
Bow_X_train = []
# calculate the dist and norm
# patch : (1 * 768) ; center : 15 * 768
# return : (1 * 15)
def dist_reciprocal(patch,center):
    a = []
    for i in range(15):
        a.append(1/np.sqrt(((patch-center[i])**2).sum()))
    a = np.array(a)
    a = a/a.sum()
    return a

# soft-max & bow train
for i in range(0,24000,16):
    arr = []
    bow_tmp = []
    for j in range(i,i+16):
        arr.append(dist_reciprocal(X_train_patches[j],bow_features))
    arr = np.array(arr)
    arr = arr.T
    bow_tmp.append(arr.max(axis = 1))
    bow_tmp = np.array(bow_tmp)
    Bow_X_train.append(bow_tmp.T)

Bow_X_train = np.array(Bow_X_train).reshape(1500,15)

# plot the problem diagram
fig = plt.figure(figsize=(10,8))
for i in range(4):
    x = np.array(range(1,16,1))
    plt.subplot(2,4,1+i)
    plt.imshow(X_train[i*375,:,:,::-1])
    plt.subplot(2,4,5+i)
    plt.bar(x,Bow_X_train[i*375])
plt.savefig('./p3_bow_hist.png')

fig = plt.figure(figsize=(10,8))
for i in range(4):
    x = np.array(range(1,16,1))
    plt.subplot(2,4,1+i)
    plt.imshow(X_train[i*375+2,:,:,::-1])
    plt.subplot(2,4,5+i)
    plt.bar(x,Bow_X_train[i*375+2])
plt.savefig('./p3_bow_hist_1.png')

fig = plt.figure(figsize=(10,8))
for i in range(4):
    x = np.array(range(1,16,1))
    plt.subplot(2,4,1+i)
    plt.imshow(X_train[i*375+10,:,:,::-1])
    plt.subplot(2,4,5+i)
    plt.bar(x,Bow_X_train[i*375+10])
plt.savefig('./p3_bow_hist_2.png')

fig = plt.figure(figsize=(10,8))
for i in range(4):
    x = np.array(range(1,16,1))
    plt.subplot(2,4,1+i)
    plt.imshow(X_train[i*375+30,:,:,::-1])
    plt.subplot(2,4,5+i)
    plt.bar(x,Bow_X_train[i*375+30])
plt.savefig('./p3_bow_hist_3.png')

#################################################
##                kNN trainning
#################################################
print("-"*20)
print("Start training kNN model...")
#trainning kNN model
label = ["banana","fountain","reef","tractor"]
Bow_training_label = np.array([label[i] for i in range(4) for j in range(375)])
kNN = KNN(n_neighbors = 5)
kNN.fit(Bow_X_train,Bow_training_label)

#testing_data setting
bow_features = k_means_center  #(15 * 768)
X_test_patches = X_test_patches  #(8000 * 768)
Bow_X_test = [] #(500 * 15)
for i in range(0,8000,16):
    arr = []
    bow_tmp = []
    for j in range(i,i+16):
        arr.append(dist_reciprocal(X_test_patches[j],bow_features))
    arr = np.array(arr)
    arr = arr.T
    bow_tmp.append(arr.max(axis = 1))
    bow_tmp = np.array(bow_tmp)
    Bow_X_test.append(bow_tmp.T)
Bow_X_test = np.array(Bow_X_test).reshape(500,15)
Bow_testing_label = np.array([label[i] for i in range(4) for j in range(125)])

# start predict data
y_pred = kNN.predict(Bow_X_test)
#calculate accuracy
count = 0.
for i in range(y_pred.shape[0]):
    if(y_pred[i] == Bow_testing_label[i]):
        count += 1
print("total accuracy : %f " %(count/y_pred.shape[0]))

count = 0.
for i in range(125):
    if(y_pred[i] == Bow_testing_label[i]):
        count += 1
print("banana accuracy : %f " %(count/(y_pred.shape[0]/4)))

count = 0.
for i in range(125,250):
    if(y_pred[i] == Bow_testing_label[i]):
        count += 1
print("fountain accuracy : %f " %(count/(y_pred.shape[0]/4)))

count = 0.
for i in range(250,375):
    if(y_pred[i] == Bow_testing_label[i]):
        count += 1
print("reef accuracy : %f " %(count/(y_pred.shape[0]/4)))

count = 0.
for i in range(375,500):
    if(y_pred[i] == Bow_testing_label[i]):
        count += 1
print("tractor accuracy : %f " %(count/(y_pred.shape[0]/4)))
