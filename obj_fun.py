import numpy as np
from Evaluation import evaln
from prac import Reg_Grow_Process
from scipy import ndimage
from scipy.stats import entropy
from Features_Extraction import weber
from GLCM import Image_GLCM
import cv2

def obj_fun(soln,inp,tar):
    if soln.ndim == 2:
        dim = soln.shape[1]
        v = soln.shape[0]
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0]; v = 1
        fitn = np.zeros((1, 1))

    for i in range(v):
        soln = np.array(soln)

        if soln.ndim == 2:
            sol = soln[i,:]
        else:
            sol = soln

        Reg_image = Reg_Grow_Process(tar,inp,np.round(sol))
        E = entropy(Reg_image)
        V = ndimage.variance(Reg_image)
        lm = V + (1/E)   # Minimization of variance and Maximization of Entropy
        fitn[i] = np.mean(lm)
    return fitn


    #grayscale_image =  cv2.cvtColor(tar[0], cv2.COLOR_BGR2GRAY)
    # WLD = weber(grayscale_image)
    # GLCM = Image_GLCM(grayscale_image)
    #
    # img = cluster.astype('uint8') * 50
    # E = entropy(img)
    # V = ndimage.variance(img)
    # lm = V + (1/E)   # Minimization of variance and Maximization of Entropy
    # fitn[i] = np.mean(lm)


