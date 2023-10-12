import cv2
import os
import numpy as np
from prac import Reg_Grow_Process
from random import uniform
from PSO import PSO
from GWO import GWO
from WOA import WOA
from EFO import EFO
from Model import Model
from Image_Results import Plot_Image
from Plot_Results import Plot_Result

##### Image read , Preprocessing and Segmentation
an = 0
if an == 1:
    filename = 'PH2Dataset/PH2 Dataset images'
    info = os.listdir(filename)  # Get the files from Dataset1
    Reg_Im = [];
    Org_Img = [];
    GT_Image = []
    for i in range(len(info)):  # For all files in Dataset1
        print(i)
        path1 = './PH2Dataset/PH2 Dataset images/' + info[i] + '/' + info[i] + '_Dermoscopic_Image' + '/' + info[
            i] + '.bmp'  # Path of the folder
        path2 = './PH2Dataset/PH2 Dataset images/' + info[i] + '/' + info[i] + '_lesion' + '/' + info[
            i] + '_lesion' + '.bmp'
        Img = cv2.imread(path1)  # Read Images
        GT = cv2.imread(path2)  # Read Images

        dim = (256, 256)
        reze_img = cv2.resize(Img, dim, interpolation=cv2.INTER_AREA)  # Scaling
        reze_GT = cv2.resize(GT, dim, interpolation=cv2.INTER_AREA)  # Scaling

        kernel = np.ones((15, 15), np.uint8)

        # Perform closing to remove hair and blur the image
        closing = cv2.morphologyEx(reze_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        blur = cv2.blur(closing, (15, 15))

        # Binarize the image
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Region Growing
        Reg_image = Reg_Grow_Process(reze_img, thresh, 3)
        Org_Img.append(reze_img)
        GT_Image.append(reze_GT)
        Reg_Im.append(Reg_image)
    np.save('Org_Img.npy', Org_Img)
    np.save('GT_Image.npy', GT_Image)
    np.save('Reg_Im.npy', Reg_Im)

##### Optimization
an = 0
if an == 1:
    Org_Img = np.load('Org_Img.npy', allow_pickle=True)
    GT_Image = np.load('GT_Image.npy', allow_pickle=True)
    sols = []
    for i in range(Org_Img.shape[0]):  # For all files in Dataset1
        Npop = 10  # Population Size
        Ch_len = 1  # Ch-length(User Define)
        xmin = np.matlib.repmat(np.concatenate([1], axis=None), Npop, 1)  # Minimum Limit
        xmax = np.matlib.repmat(np.concatenate([5], axis=None), Npop, 1)  # Maximum Limit
        initsol = np.zeros((xmax.shape))  # Initial Solution
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
        fname = 'obj_fun'  # Function Name
        Max_iter = 25  # Maximum Iterations

        print("PSO...")
        [bestfit1, fitness1, bestsol1, time1] = PSO(initsol, fname, xmin, xmax, Max_iter, Org_Img[i],
                                                    GT_Image[i])  # PSO

        print("WOA...")
        [bestfit2, fitness2, bestsol2, time2] = WOA(initsol, fname, xmin, xmax, Max_iter, Org_Img[i],
                                                    GT_Image[i])  # WOA

        print("EFO...")
        [bestfit3, fitness3, bestsol3, time3] = EFO(initsol, fname, xmin, xmax, Max_iter, Org_Img[i],
                                                    GT_Image[i])  # EFO

        print("GWO...")
        [bestfit4, fitness4, bestsol4, time4] = GWO(initsol, fname, xmin, xmax, Max_iter, Org_Img[i],
                                                    GT_Image[i])  # GWO

        Bestsol = [bestsol1, bestsol2, bestsol3, bestsol4]
        sols.append(Bestsol)
    np.save('sols.npy', sols)  # Save the Best Solutions

##### Classification
an = 0
if an == 1:
    sols = np.load('sols.npy', allow_pickle=True)
    Feat = np.load('Feat.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    all_Eval = []
    each = []
    for i in range(len(learnper)):
        learnperc = round(Target.shape[0] * learnper[i])
        Eval = np.zeros((12, 14))
        Each = np.zeros((12, Target.shape[1], 14))
        for j in range(4):
            sol = np.round(Bestsol[j, :])
            Train_Data = Feat[:learnperc, :, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :, :]
            Test_Target = Target[learnperc:, :]

            Eval[j, :] = Model(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        all_Eval.append(Eval)
    np.save('all_Eval.npy', all_Eval)  # Save the Best Solutions

# Plot the Desired Results
Plot_Image()  # Output Images
Plot_Result()  # Graph Results
