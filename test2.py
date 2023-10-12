import cv2
from seg import Region_Growing
import os

def region_growing(image_data, neighbours, threshold=10, segmentation_name="Region Growing"):
	region_growing = Region_Growing(image_data, threshold=threshold, conn=neighbours)
	# Set Seeds
	region_growing.set_seeds()
	# Segmentation
	region_growing.segment()
	# Display Segmentation
	region_growing.display_and_resegment(name=segmentation_name)

CONN = 4
filename = 'PH2Dataset/PH2 Dataset images'
info = os.listdir(filename)  # Get the files from Dataset1
path = './PH2Dataset/PH2 Dataset images/' + info[0] + '/' + info[0] + '_Dermoscopic_Image' + '/' + info[0] + '.bmp'
image_data = cv2.imread(path)

if image_data.shape[0] > 1000:
	image_data = cv2.resize(image_data, (0, 0), fx=0.25, fy=0.25)
if image_data.shape[0] > 500:
	image_data = cv2.resize(image_data, (0, 0), fx=0.5, fy=0.5)

image_data_post_smoothing = cv2.GaussianBlur(image_data,(3,3),0)

region_growing(image_data_post_smoothing, segmentation_name=" segmentation", neighbours=CONN)






