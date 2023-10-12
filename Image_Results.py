import cv2
import os
import numpy as np
from prac import Reg_Grow_Process

def Plot_Image():
    an = 0
    if an == 1:   # Image Write
        filename = 'PH2Dataset/PH2 Dataset images'
        info = os.listdir(filename)   # Get the files from Dataset1
        for i in range(3):  # For all files in Dataset1
            path1 = './PH2Dataset/PH2 Dataset images/'+info[i]+'/'+info[i]+'_Dermoscopic_Image' + '/' + info[i]+'.bmp'  # Path of the folder
            path2 = './PH2Dataset/PH2 Dataset images/'+info[i]+'/'+info[i]+'_lesion' + '/' +info[i]+'_lesion'  + '.bmp'
            Img = cv2.imread(path1)   # Read Images
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
            Reg_image = Reg_Grow_Process(reze_img,thresh)

            name1 = './Results/orgi_%s.jpg' % (i)
            cv2.imwrite(name1, reze_img)

            name2 = './Results/GT_%s.jpg' % (i)
            cv2.imwrite(name2, reze_GT)

            name3 = './Results/Pre_%s.jpg' % (i)
            cv2.imwrite(name3, thresh)

            name4 = './Results/Reg_grow_%s.jpg' % (i)
            cv2.imwrite(name4, Reg_image)
    else:       # Display The Images
        for i in range(3):  # For all files in Dataset1
            n1 = './Results/orgi_%s.jpg' % (i)
            name1 = cv2.imread(n1)
            cv2.imshow("reze_org_img",name1)

            n2 = './Results/GT_%s.jpg' % (i)
            name2 = cv2.imread(n2)
            cv2.imshow("reze_GT",name2)

            n3 = './Results/Pre_%s.jpg' % (i)
            name3 = cv2.imread(n3)
            cv2.imshow("thresh",name3)

            n4 = './Results/Reg_grow_%s.jpg' % (i)
            name4 = cv2.imread(n4)
            cv2.imshow("Reg_grow_img",name4)
            cv2.waitKey(0)

