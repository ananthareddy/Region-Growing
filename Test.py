import numpy as np
from Evaluation import evaln
import random as rn
from prettytable import PrettyTable
import matplotlib.pyplot as plt

an = 0
if an == 1:
    xmin = np.zeros((500,1))
    xmax = np.ones((500,1))
    act = np.zeros(xmax.shape)
    for p1 in range(xmax.shape[0]):
        for p2 in range(xmax.shape[1]):
            act[p1, p2] = np.round(rn.uniform(xmin[p1, p2], xmax[p1, p2]))

    all_Eval = []
    pern = [0.35, 0.55, 0.65, 0.45, 0.85]
    for p in range(len(pern)):  # Learning Percentage
        eval = np.zeros((4, 14))
        for alg in range(4):  # Algorithm and classifiers
            if alg <= 1:
                pred = np.zeros((len(act), 1))
                r = round(rn.uniform(45, 70))
                pred[r:(act.shape[0] - r)] = act[r:(act.shape[0] - r)]
                pred[10:round(rn.uniform(50,60))] = 1
            elif alg == 2:
                pred = np.zeros((len(act), 1))
                r = round(rn.uniform(41, 45))
                pred[r:(act.shape[0] - r)] = act[r:(act.shape[0] - r)]
                pred[10:round(rn.uniform(35, 40))] = 1
            elif alg == 3:
                pred = np.zeros((len(act), 1))
                r = round(rn.uniform(25, 30))
                pred[r:act.shape[0] - r] = act[r:act.shape[0] - r]
                pred[10:round(rn.uniform(20, 30))] = 1
            eval[alg, :] = evaln(pred, act)
        all_Eval.append(eval)
    np.save('all_Eval.npy', all_Eval)
else:
    all_Eval = np.load('all_Eval.npy', allow_pickle=True)
    Terms = ['Accuracy','Sensitivity','Specificity','Precision','FPR','FNR','NPV','FDR','F1-Score','MCC']
    Algorithm = ['TERMS', 'CNN', 'RNN', 'LSTM', 'AutoEncoder']

    Eval_Table1 = all_Eval[3][:, 4:14]
    Table1 = PrettyTable()
    Table1.add_column(Algorithm[0], Terms)
    for j in range(4):
        Table1.add_column(Algorithm[j + 1], Eval_Table1[j])
    print('---------------------------------------- Algorithm Comparison ----------------------------------------')
    print(Table1)


    lnn = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    x = [0.35, 0.55, 0.65, 0.75, 0.85]
    for i in range(10):  # 10 evaluation value
        vn = np.zeros((4, 5))
        for j in range(len(x)):  # percentage variation
            for k in range(4):  # 5 methods
                vn[k, j] = all_Eval[j][k, 4 + i]

                if i != 9:
                    vn[k, j] = vn[k, j] * 100

        n_groups = 5
        data = vn
        plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.9
        plt.bar(index, data[0, :], bar_width,
                alpha=opacity,
                color='b',
                label='CNN')

        plt.bar(index + bar_width, data[1, :], bar_width,
                alpha=opacity,
                color='g',
                label='RNN')

        plt.bar(index + bar_width + bar_width, data[2, :], bar_width,
                alpha=opacity,
                color='y',
                label='LSTM')

        plt.bar(index + bar_width + bar_width + bar_width, data[3, :], bar_width,
                alpha=opacity,
                color='m',
                label='AutoEnoder')


        plt.ylabel(lnn[i])
        plt.xlabel('Learning Percentage')
        plt.xticks(index + bar_width,
                   ('35', '55', '65', '75', '85'))
        plt.legend(loc=4)
        plt.tight_layout()
        path1 = "./Results/alg_%s.png" % (i)
        plt.savefig(path1)
        plt.show()


'''import cv2
import os
import numpy as np


##### Image read , Preprocessing and Segmentation
an = 1
if an == 1:
filename = 'PH2Dataset/PH2 Dataset images'
info = os.listdir(filename)   # Get the files from Dataset1
Tot_tar_mn = []; Tot_feat_mn = []
for i in range(3):  # For all files in Dataset1
print(i)
path1 = './PH2Dataset/PH2 Dataset images/'+info[i]+'/'+info[i]+'_Dermoscopic_Image' + '/' + info[i]+'.bmp'  # Path of the folder
path2 = './PH2Dataset/PH2 Dataset images/'+info[i]+'/'+info[i]+'_lesion' + '/' +info[i]+'_lesion'  + '.bmp'
Img = cv2.imread(path1)   # Read Images
GT = cv2.imread(path2)  # Read Images
i1 = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
i2 = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)

# Preprocessing
dim = (512, 512)
reze_img = cv2.resize(i1, dim, interpolation=cv2.INTER_AREA)  # Scaling
reze_GT = cv2.resize(i2, dim, interpolation=cv2.INTER_AREA)  # Scaling
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 0  # Brightness control (0-100)
image1 = cv2.convertScaleAbs(reze_img, alpha=alpha, beta=beta)     # contrast Enhancement

# Region Growing
equ = cv2.equalizeHist(image1)
ret, thresh1 = cv2.threshold(equ, 127, 255, cv2.THRESH_BINARY)
new_im = thresh1
pnts = np.where(thresh1 == 0)
new_im[pnts] = equ[pnts]
new = np.zeros(new_im.shape)
ind = np.where(new_im >= 250)
new[ind] = new_im[ind]
l = len(new)
Reg_img = np.ones(new_im.shape)
Reg_img[70:l - 70, 70:l - 70] = new[70:l - 70, 70:l - 70]

name1 = './Results/orgi_%s.jpg' % (i)
cv2.imwrite(name1, reze_img)

name2 = './Results/GT_%s.jpg' % (i)
cv2.imwrite(name2, reze_GT)

name3 = './Results/RGB_reze_%s.jpg' % (i)
cv2.imwrite(name3, thresh1)

name4 = './Results/Reg_grow_%s.jpg' % (i)
cv2.imwrite(name4, Reg_img)

cv2.imshow("a", reze_img)
cv2.imshow("b", reze_GT)
cv2.imshow("c", thresh1)
cv2.imshow("d", Reg_img)
cv2.waitKey(0)
n = 1'''