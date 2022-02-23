import numpy as np
from sklearn.decomposition import PCA
import imageio
import os
from matplotlib import pyplot as plt
from random import randint, shuffle
import logging as lg
lg.basicConfig(level=lg.WARNING, format='%(levelname)s | %(message)s')
logger = lg.getLogger(__name__)

# Set logging level for our file
logger.setLevel(lg.INFO)

plt.rcParams["figure.figsize"] = (14,20)

def read_img(path):
    """Loads a single image into a 3 channel RGB numpy array"""
    im = imageio.imread(path)
    if im.shape[2] == 4:
        im = np.delete(im, [3], 2)
    im = im.astype(np.int32)
    return im.squeeze()

# ---=== EDIT THIS TO CHANGE THE SOURCE FOLDER ===---
FOLDER = "sample_strips"

# Load every image in the specified folder
images = []
for f in list(os.listdir(FOLDER)):
    images.append(read_img(os.path.join(FOLDER, f)))
shuffle(images)
images = np.array(images)
logger.info(f"Successfully loaded {len(images)} images")

# Create a list of all colours in the images
count, height, channels = images.shape
colours = images.reshape(count * height, channels)

# Extract the two principle components of the colours - greyness and blueness
pca = PCA(n_components=2)
pca.fit(colours)

# Transform the images into this new colour space
transformed_images = np.zeros((count, height, 2))
for i in range(len(images)):
    transformed_images[i] = pca.transform(images[i])
images = transformed_images

# If set to debug, display the strip images
if lg.DEBUG >= logger.level:
    fig, axs = plt.subplots(2)
    for channel in [0,1]:
        axs[channel].imshow(transformed_images[:,:,channel].T, cmap="gray_r")
        axs[channel].set_title(f"Raw strips (channel PC{channel})")
    plt.show()

# Function to determine how closely matched two strips are
# This is likely what you want to be messing around with
def similarity(a,b): 
    header_score = np.sum(100 / (np.linalg.norm(a[:230]-b[:230], axis=1) + 1))
    text_score = np.sum(100 / (np.linalg.norm(a[230:]-b[230:], axis=1) + 1))
    return header_score*2 + text_score

logger.debug(f"The similarity of the first two images is {similarity(images[0], images[1])}")

# Calculate comparison scores for each pair of strips
logger.info(f"Calculating scores for {(len(images)*len(images)-1)//2} pairs of strips")
sims = {}
for i, im in enumerate(images):
    for j, im2 in enumerate(images):
        if j <= i: continue
        sims[(i,j)] = similarity(im, im2)
    if i % 250 == 0:
        logger.info(f"Done {i}/{len(images)} images")
logger.info("Complete!")

# Build the image based on the similarity matrix
START_POINT = randint(0, len(images)-1)
build = [START_POINT]
remaining = set(range(len(images)))
remaining.remove(START_POINT)

logger.info("Assembling image...")
# Each step, take the best matching strip and add it to the build
while len(remaining) > 0:
    best_score = -1000000000
    best_idx = None
    for idx2 in remaining:
        i = min(build[-1], idx2)
        j = max(build[-1], idx2)
        if sims[(i,j)] > best_score:
            best_score = sims[(i,j)]
            best_idx = idx2
    logger.debug(f"{len(build)}) Best Score was {best_score} at index {best_idx}")
    build.append(best_idx)
    remaining.remove(best_idx)
logger.info("Assembling Complete!")

# Actually concatenate the image data together
logger.info("Image exported to ./export.png")
pca.inverse_transform(images[0])
built_image = np.concatenate([np.expand_dims(pca.inverse_transform(images[i]),1).astype(np.uint8) for i in build], axis=1)
imageio.imwrite("export.png", built_image.astype(np.uint8))
plt.imshow(built_image, cmap='gray')
plt.show()