import numpy as np
from scipy.signal import convolve2d
#from skimage.feature import graycomatrix, graycoprops


def weber(grayscale_image):
    grayscale_image = grayscale_image.astype(np.float64)
    grayscale_image[grayscale_image==0] = np.finfo(float).eps
    neighbours_filter = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ])
    convolved = convolve2d(grayscale_image,neighbours_filter, mode='same')
    weber_descriptor = convolved-8*grayscale_image
    weber_descriptor = weber_descriptor/grayscale_image
    weber_descriptor = np.arctan(weber_descriptor)
    return weber_descriptor



'''def GLCM_feat(image):
    PATCH_SIZE = 21

    # select some patches from grassy areas of the image
    grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
    grass_patches = []
    for loc in grass_locations:
        grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                   loc[1]:loc[1] + PATCH_SIZE])

    # select some patches from sky areas of the image
    sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
    sky_patches = []
    for loc in sky_locations:
        sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])

    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (grass_patches + sky_patches):
        glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(graycoprops(glcm, 'correlation')[0, 0])

    return xs,ys'''



