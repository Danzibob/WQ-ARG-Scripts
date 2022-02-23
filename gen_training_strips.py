import numpy as np
import imageio
import os
import matplotlib

def read_img(path):
    im = imageio.imread(path).squeeze()
    return im

FOLDER = "sample_pages"
OUT_FOLDER = "sample_strips"

size = (10000, 10000)

if __name__ == "__main__":
    for f in list(os.listdir(FOLDER)):
        im = read_img(os.path.join(FOLDER,f))
        x_size, y_size, channels = im.shape
        size = (min(size[0], x_size), min(size[1], y_size))
    print(f"Cropping images to size {size}")
    for f in list(os.listdir(FOLDER)):
        im = read_img(os.path.join(FOLDER,f))
        im = im[:size[0], :size[1], :]
        for col in range(im.shape[1]):
            strip = im[:,col:col+1]
            imageio.imwrite(os.path.join(OUT_FOLDER, f"{f[:-4]}-{col}.png"), strip.astype(np.uint8))