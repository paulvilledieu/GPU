import numpy as np
import cv2
import sys
import pandas as pd

def image_to_txt(src, dst):
    im = cv2.imread(src, cv2.IMREAD_GRAYSCALE).astype(int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] < 127:
                im[i, j] = 0
            else:
                im[i, j] = 255
    np.savetxt(dst, im, fmt='%i')

def txt_to_image(src, dst):
    image = pd.read_csv(src, sep=' ').values
    cv2.imwrite(dst, image)

if len(sys.argv) < 3:
    "Please provide a source, a dest."
else:
    txt_to_image(sys.argv[1], sys.argv[2])
    #image_to_txt(sys.argv[1], sys.argv[2])