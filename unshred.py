import numpy as np
from sklearn.decomposition import PCA
import imageio
import os, json
from matplotlib import pyplot as plt
from random import randint, shuffle, choice
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
FOLDER = "strips"

# Load json mapping for image IDs
if FOLDER == "strips":
    with open("mapping.json") as f:
        mapping = json.load(f)

# Load every image in the specified folder
keys = []
images = {}
for f in list(os.listdir(FOLDER)):
    if FOLDER == "strips":
        key = mapping[f[:-6]] + f[-6:-4]
    else:
        key = f
    keys.append(key)
    images[key] = read_img(os.path.join(FOLDER, f))
logger.info(f"Successfully loaded {len(images)} images")

# Create a list of all colours in the images
colours = np.concatenate(list(images.values()))

# Extract the two principle components of the colours - greyness and blueness
pca = PCA(n_components=2)
pca.fit(colours)

# Transform the images into this new colour space
for k, v in images.items():
    images[k] = pca.transform(v)

# If set to debug, display the strip images
if lg.DEBUG >= logger.level:
    fig, axs = plt.subplots(2)
    for channel in [0,1]:
        axs[channel].imshow(np.array(list(images.values()))[:,:,channel].T, cmap="gray_r")
        axs[channel].set_title(f"Raw strips (channel PC{channel})")
    plt.show()

# Function to determine how closely matched two strips are
# This is likely what you want to be messing around with
def similarity(a,b): 
    header_score = np.sum(100 / (np.linalg.norm(a[:230]-b[:230], axis=1) + 1))
    text_score = np.sum(100 / (np.linalg.norm(a[230:]-b[230:], axis=1) + 1))
    return header_score*2 + text_score

logger.debug(f"The similarity of the first two images is {similarity(images[keys[0]], images[keys[1]])}")

# Calculate comparison scores for each pair of strips
logger.info(f"Calculating scores for {(len(images)*len(images)-1)//2} pairs of strips")


JSON_CACHE_PATH = FOLDER + "-similarity_cache.json"
if os.path.exists(JSON_CACHE_PATH):
    logger.info("Found existing similarity cache, loading...")
    with open(JSON_CACHE_PATH) as f:
        sims = json.load(f)
else:
    logger.info("No similarity cache found, building from scratch...")
    sims = {}

for i, k1 in enumerate(keys):
    for k2 in keys[i+1:]:
        sim_key = min(k1, k2) + "," + max(k1,k2)
        if sim_key in sims: continue
        sims[sim_key] = similarity(images[k1], images[k2])
    if i % 250 == 0:
        logger.info(f"Done {i}/{len(images)} images")
logger.info("Complete!")

with open(JSON_CACHE_PATH, 'w') as outfile:
    json.dump(sims, outfile)

logger.info(f"Saved cache at {JSON_CACHE_PATH}")

# Build the image based on the similarity matrix
START_POINT = choice(keys)
build = [START_POINT]
remaining = set(keys)
remaining.remove(START_POINT)
logger.info("Assembling image...")
logger.debug(f"Starting with key {START_POINT}")
# Each step, take the best matching strip and add it to the build
while len(remaining) > 0:
    best_score = -1000000000
    best_idx = None
    k1 = build[-1]
    for k2 in remaining:
        sim_key = min(k1, k2) + "," + max(k1,k2)
        if sims[sim_key] > best_score:
            best_score = sims[sim_key]
            best_idx = k2
    logger.debug(f"{len(build)}) Best Score was {best_score} at index {best_idx}")
    build.append(best_idx)
    remaining.remove(best_idx)
logger.info("Assembling Complete!")

# Actually concatenate the image data together
built_image = np.concatenate([np.expand_dims(pca.inverse_transform(images[k]),1).astype(np.uint8) for k in build], axis=1)
imageio.imwrite("export.png", built_image.astype(np.uint8))
logger.info("Image exported to ./export.png")
plt.imshow(built_image, cmap='gray')
plt.show()