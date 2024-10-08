import decimal
import pandas as pd
import os
import re
import numpy as np
import collections
from sklearn.metrics import confusion_matrix
from sklearn import mixture
from scipy.spatial import distance
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

os.chdir('\raw extracted data')

n_cluster = 3  # Number of cluster of GMM

filegroup = ['119_1_denoised.txt','120_1_denoised.txt','120_2_denoised.txt','121_2_denoised.txt','122_1_denoised.txt','123_1_denoised.txt','125_1_denoised.txt','127_1_denoised.txt','128_1_denoised.txt','128_2_denoised.txt','128_3_denoised.txt','129_1_denoised.txt','129_2_denoised.txt','130_1_denoised.txt']
file_len= [0.2,0.2,0.4,0.3,0.3,0.3,0.435,0.2,0.3,0.3,0.4,0.4,0.3,0.3]

l=[]
j=1
smooth_lines_total=[]
smooth_length_total=[]
ind=0
for filename in filegroup:
    data = pd.read_csv(filename,  sep="\t", header = None, index_col=False, float_precision= "round_trip", names=["north", "east", "height", "time", "date", "lines", "length", "green", "status"])
    z2= data.groupby(['north', 'east'], sort=False)['lines','length'].mean()
    if int(7.5/file_len[ind]) % 2 == 0:
        window_size = int(7.5/file_len[ind])+1
    else:
        window_size = int(7.5/file_len[ind])    
    poly_order = 1       
    smooth_lines = savgol_filter(z2.lines, window_size, poly_order)
    smooth_lines_total.extend(smooth_lines)
    smooth_length = savgol_filter(z2.length, window_size, poly_order)
    smooth_length_total.extend(smooth_length)
    j=j+1
    ind=ind+1
   
###############################################################################################
###############################################################################################
###############################################################################################

smooth_lines_total=np.array(smooth_lines_total).reshape(-1,1)
smooth_length_total=np.array(smooth_length_total).reshape(-1,1)
feature=np.concatenate((smooth_lines_total,smooth_length_total),axis=1)
clf=mixture.GaussianMixture(n_components=n_cluster , covariance_type='full',n_init=10)
clf.fit(feature)
cls = clf.predict(feature) 
clp=clf.predict_proba(feature)   
# extract cluster labels
cds = clf.means_        
# extract cluster centroids (means of gaussians)
covs = clf.covariances_
#fig=plt.figure(num=None, figsize=(10, 8), dpi=150, facecolor='w', edgecolor='k')
#clusterplot(feature, clusterid=cls, centroids=cds, covars=covs)

##############################################################################

#Assigning the right cluster to right coverage

origin = np.array([0,0])

if n_cluster == 3:
    sorted_mean = [16.5, 50, 83.5]
    dist1 = np.linalg.norm(origin-clf.means_[0])
    dist2 = np.linalg.norm(origin-clf.means_[1])
    dist3 = np.linalg.norm(origin-clf.means_[2])
    sorted_clusters = np.argsort([dist1, dist2, dist3])
elif n_cluster == 4:
    sorted_mean = [12.5, 37.5, 62.5, 87.5]
    dist1 = np.linalg.norm(origin-clf.means_[0])
    dist2 = np.linalg.norm(origin-clf.means_[1])
    dist3 = np.linalg.norm(origin-clf.means_[2])
    dist4 = np.linalg.norm(origin-clf.means_[3])
    sorted_clusters = np.argsort([dist1, dist2, dist3, dist4])
elif n_cluster == 5:
    sorted_mean = [10, 30, 50, 70, 90]
    dist1 = np.linalg.norm(origin-clf.means_[0])
    dist2 = np.linalg.norm(origin-clf.means_[1])
    dist3 = np.linalg.norm(origin-clf.means_[2])
    dist4 = np.linalg.norm(origin-clf.means_[3])
    dist5 = np.linalg.norm(origin-clf.means_[4])
    sorted_clusters = np.argsort([dist1, dist2, dist3, dist4, dist5])
elif n_cluster == 6:
    sorted_mean = [8.5, 25, 41.5, 58.5, 75, 91.5]
    dist1 = np.linalg.norm(origin-clf.means_[0])
    dist2 = np.linalg.norm(origin-clf.means_[1])
    dist3 = np.linalg.norm(origin-clf.means_[2])
    dist4 = np.linalg.norm(origin-clf.means_[3])
    dist5 = np.linalg.norm(origin-clf.means_[4])
    dist6 = np.linalg.norm(origin-clf.means_[5])
    sorted_clusters = np.argsort([dist1, dist2, dist3, dist4, dist5, dist6])
    


##############################################################################
# Predicting status using GMM

os.chdir('..\raw extracted data')
p62=[]
l1=[]
filegroup = ['119_1_denoised.txt','120_1_denoised.txt','120_2_denoised.txt','121_2_denoised.txt','122_1_denoised.txt','123_1_denoised.txt','125_1_denoised.txt','127_1_denoised.txt','128_1_denoised.txt','128_2_denoised.txt','128_3_denoised.txt','129_1_denoised.txt','129_2_denoised.txt','130_1_denoised.txt']
filegroup1 = ['119_1_unique.txt','120_1_unique.txt','120_2_unique.txt','121_2_unique.txt','122_1_unique.txt','123_1_unique.txt','125_1_unique.txt','127_1_unique.txt','128_1_unique.txt','128_2_unique.txt','128_3_unique.txt','129_1_unique.txt','129_2_unique.txt','130_1_unique.txt']
import csv
tp=0
sum=0
count=0
ind=0

for filename in filegroup:
    data = pd.read_csv(filename,  sep="\t", header = None, index_col=False, float_precision= "round_trip", names=["north", "east", "height", "time", "date", "lines", "length", "green", "status"])

    n=len(data)


    z1= data.groupby(['north', 'east'], sort=False)['lines'].mean()
    z3= data.groupby(['north', 'east'], sort=False)['length'].mean()
    
    z2= data.groupby(['north', 'east'], sort=False)['lines','length'].mean()

#Smoothing function
    if int(7.5/file_len[ind]) % 2 == 0:
        window_size = int(7.5/file_len[ind])+1
    else:
        window_size = int(7.5/file_len[ind])    
    poly_order = 1     

    smooth_lines = savgol_filter(z2.lines, window_size, poly_order)
    smooth_length = savgol_filter(z2.length, window_size, poly_order)
    p=[]
    p1=[]

    for i in range(len(smooth_lines)):

        #index=np.where(sorted_clusters==clf.predict(np.asarray((smooth_lines[i], smooth_length[i])).reshape(1,-1))[0])[0]
        prob = clf.predict_proba(np.asarray((smooth_lines[i], smooth_length[i])).reshape(1,-1))[0]
        p.append(np.dot(prob[sorted_clusters],sorted_mean))


    l1.append(len(smooth_lines))
    print(ind, len(smooth_lines))
    
    np.savetxt(list(zip(z1.index,p)), fmt='%s\t')

    count=count+1
    ind=ind+1
