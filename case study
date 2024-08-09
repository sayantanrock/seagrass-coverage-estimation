import os
import cv2
import numpy as np
from pylsd import lsd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# web link for the dataset : https://github.com/EnviewFulda/LookingForSeagrass?tab=readme-ov-file 
# PAPER : G. Reus et al., "Looking for Seagrass: Deep Learning for Visual Coverage Estimation," 2018 OCEANS - MTS/IEEE Kobe Techno-Oceans (OTO), Kobe, Japan, 2018, pp. 1-6, doi: 10.1109/OCEANSKOBE.2018.8559302.

path = 'D:\\seagrass-dataset\\dataset\\images\\01d02\\'          # PATH FOR RAW IMAGES
path2 = 'D:\\seagrass-dataset\\dataset\\ground-truth\\01d02\\'   # PATH FOR GROUND TRUTH ANNOTATED IMAGES


for filename in os.listdir(path):
    path1=path+filename
    img = cv2.imread(path1,1)
    img1 = cv2.imread(path1,0)
    #file1 = filename.split(".")[1]
    file1 = filename.split(".jpg")[0]
    #cv2.imshow('image',img)
    
    new_path = path2 + 'pm_' + file1 + '.png'
    img3 = cv2.imread(new_path,0)
    img4 = cv2.imread(new_path,1)
    img5 = np.pad(img3,((0,1),(0,1)), 'edge')

    a=[]
    tot_dist=0
    count=0
    black_count = 0
    lines = lsd(img1)
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 1]), int(lines[i, 0]))
        pt2 = (int(lines[i, 3]), int(lines[i, 2]))
        pt3 = (int(lines[i, 0]), int(lines[i, 1]))
        pt4 = (int(lines[i, 2]), int(lines[i, 3]))
        
        width = lines[i, 4]
        
        count=count+1
        dst = distance.euclidean(pt1,pt2)
        tot_dist = tot_dist + dst
        
        count1 = (0,1)[img5[pt2] == 0 or img5[pt1] == 0]
        black_count = black_count+count1
        cv2.line(img, pt3, pt4, (0, 0, 255), int(np.ceil(width / 2)))
        cv2.line(img4, pt3, pt4, (0, 0, 255), int(np.ceil(width / 2)))
    
    ratio_black_pixels = (np.shape(img3)[0]*np.shape(img3)[1] - np.count_nonzero(img3))/(np.shape(img3)[0]*np.shape(img3)[1])
    f = open('D:\\seagrass-dataset\\dataset\\lsd_text_new\\01d02.txt', 'a')                                                              # SAVE EXTRACTED FEATURES IN A FILE
    f.write('%s\t%d\t%d\t%.2f\t%.2f\n' % (file1, count, black_count, ratio_black_pixels, tot_dist))
    f.close()

    cv2.imwrite('D:\\seagrass-dataset\\dataset\\lsd\\01d02\\'+file1+'_lsd_ground.jpg',img4)     # SAVE EXTRACTED FEATURES IN A IMAGE
    cv2.imwrite('D:\\seagrass-dataset\\dataset\\lsd\\01d02\\'+file1+'_lsd.jpg',img)
#    cv2.imshow('frame',img4)
#    cv2.waitKey(0)
    cv2.destroyAllWindows()


import pandas as pd
data = pd.read_csv('D:\\seagrass-dataset\\dataset\\lsd_text_new\\01d02.txt',sep='\t',header=None,index_col=False, float_precision= "round_trip",names=["file","lines","black_lines","ratio","length"])
fig=plt.figure(num=None, figsize=(10, 8), dpi=150, facecolor='w', edgecolor='k')
plt.title('Lines vs Length')
plt.xlabel('Percentage of Covered Vegetation (pixelwise)')
plt.ylabel('Total Number of Lines')
plt.scatter( data.ratio, data.black_lines)
plt.savefig('D:\\seagrass-dataset\\dataset\\test2.jpg')


from sklearn.linear_model import LinearRegression
y=data.ratio
X=data[["black_lines"]]
pr_model = LinearRegression()


import statsmodels.api as sm
import numpy as np
from numpy.random import uniform, normal, poisson, binomial
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit


exog, endog = sm.add_constant(X), y
mod = sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.logit()))
#mod = sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.logit()))
res = mod.fit()
display(res.summary())
y_pred = res.predict(exog)

fig=plt.figure(num=None, figsize=(10, 8), dpi=150, facecolor='w', edgecolor='k')
plt.scatter(X, y_pred, color='m')
plt.scatter(X, y,  s=20, alpha=0.28)
plt.plot([260, 260], [0, 1], 'r--', lw=2)
plt.plot([0, len(X)], [0.37, 0.37], 'g--', lw=2)
plt.xlabel("Number of Lines detected in seagrass annotated region")
plt.ylabel("Ratio of the number of pixels annotated as seagrass in an image")
plt.savefig('D:\\seagrass-dataset\\dataset\\logit_cutoff.jpg')



#comparing quantificaton method on reus
X=data[["lines"]]
smooth_lines_total=np.asarray(X)
X=data[["black_lines"]]
smooth_lines_black=np.asarray(X)

smooth_new=  smooth_lines_black/smooth_lines_total
fig=plt.figure(num=None, figsize=(10, 8), dpi=150, facecolor='w', edgecolor='k')
plt.hist(smooth_lines_total, bins=50)
plt.xlabel("Total number of Lines")
plt.savefig('D:\\seagrass-dataset\\dataset\\histogram.jpg')

lower_quantile_lines = np.percentile(smooth_lines_total,10)
upper_quantile_lines = np.percentile(smooth_lines_total,95)
smooth_lines_new = (smooth_lines_total -  lower_quantile_lines)/(upper_quantile_lines-lower_quantile_lines)
smooth_lines_new[smooth_lines_new < 0] = 0
smooth_lines_new[smooth_lines_new > 1] = 1

fig=plt.figure(num=None, figsize=(10, 8), dpi=150, facecolor='w', edgecolor='k')
plt.scatter(logit(y),logit(snmooth_new), color='m')
#plt.scatter(X, y,  s=20, alpha=0.28)
#plt.plot([260, 260], [0, 1], 'r--', lw=2)
#plt.plot([0, len(X)], [0.37, 0.37], 'g--', lw=2)
plt.xlabel("logit(Ratio of seagrass pixels in the Image)")
plt.ylabel("logit(Ratio of total lines in black region in each image)")
plt.savefig('D:\\seagrass-dataset\\dataset\\quantification_logit.jpg')
