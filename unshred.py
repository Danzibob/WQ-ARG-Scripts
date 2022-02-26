import numpy as np
from sklearn.decomposition import PCA
import imageio
import os, json
from matplotlib import pyplot as plt
from random import randint, shuffle, choice
import logging as lg
from multiprocessing import Pool
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
    # with open("unclustered.json") as f:
    #     arrangement = json.load(f)

# Load every image in the specified folder
keys = []
images = {}
for f in list(os.listdir(FOLDER)):
    if FOLDER == "strips":
        key = mapping[f[:-6]] + f[-6:-4]
        # if key in arrangement.keys(): 
        keys.append(key)
        images[key] = read_img(os.path.join(FOLDER, f))
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

# Log common colours
ref_cols = pca.transform([[0,0,0], [255,255,255]])
logger.info(f"PCA Results: Black -> {ref_cols[0]} | White -> {ref_cols[1]}")

# We care more about black matching black than white matching white
# Therefore whites should be closer to eachother than blacks

# Transform the images into this new colour space
assert ref_cols[0][0] > ref_cols[0][1]
MIN_WHITE = ref_cols[1][0]*-1
MAX_BLACK = ref_cols[0][0]
print(f"Min white: {MIN_WHITE}, Max Black: {MAX_BLACK}")
for k, v in images.items():
    images[k] = pca.transform(v)
    images[k][:,0] = np.square((images[k][:,0]+MIN_WHITE))/MAX_BLACK
    images[k][:,1] *= 4

# If set to debug, display the strip images
if lg.DEBUG >= logger.level:
    fig, axs = plt.subplots(2)
    for channel in [0,1]:
        axs[channel].imshow(np.array(list(images.values()))[:,:,channel].T, cmap="gray_r")
        axs[channel].set_title(f"Raw strips (channel PC{channel})")
    plt.show()

# Function to determine how closely matched two strips are
# This is likely what you want to be messing around with
SOBEL = np.array([-1, 0, 1])
def edge_detect(x):
    h, chans = x.shape
    new_image = np.zeros((h+2, chans))
    for c in range(chans):
        new_image[:,c] = np.convolve(x[:,c], SOBEL)
    return new_image

def similarity(a,b):
    score = 0
    for roll, weight in zip((-1,0,1),(0.5,1,0.5)):
        score += np.mean(100 / (np.linalg.norm(a-np.roll(b,roll), axis=1) + 1))*weight
    return score/2

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

cached_keys = sims.keys()

todo = set()
for i, k1 in enumerate(keys):
    for k2 in keys[i+1:]:
        sim_key = min(k1, k2) + "," + max(k1,k2)
        if sim_key not in cached_keys: todo.add(sim_key)
    if i % 250 == 0:
        logger.info(f"Scheduling {i}/{len(images)} images")
logger.info("Done scheduling!")

if len(todo) > 0:
    THREADS = 8
    from tqdm import tqdm
    pbar = tqdm(total=len(todo))

    def get_sim(k1k2):
        k1, k2 = k1k2.split(",")
        score = similarity(images[k1], images[k2])
        pbar.update(THREADS)
        return score

    from time import time
    logger.info(f"Calculating similarity for {len(todo)} pairs")
    if len(todo) > 800000:
        logger.info("This may take some time...")
    start = time()
    pool = Pool(THREADS)
    out = pool.map(get_sim, todo)

    scores = out
    logger.info(f"Building Dictionary")
    kv = dict(zip(todo, scores))
    logger.info(f"Merging Dictionary")
    sims = {**sims, **kv}
    t = time() - start
    logger.info(f"Took {t} seconds!")

    with open(JSON_CACHE_PATH, 'w') as outfile:
        json.dump(sims, outfile)

    logger.info(f"Saved cache at {JSON_CACHE_PATH}")

# Build the image based on the similarity matrix
START_POINT = choice(keys)
build = [START_POINT]
scores = []
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
    scores.append(best_score)
    remaining.remove(best_idx)
logger.info("Assembling Complete!")

score_edges = np.square(np.convolve(scores, np.array([-1,-1,-1,1,1,1]), 'valid'))/10

for k in images:
    # images[k][:,0] = np.square((images[k][:,0]+MIN_WHITE))/MAX_BLACK
    images[k][:,0] = np.sqrt(images[k][:,0]*MAX_BLACK)-MIN_WHITE
    # images[k][:,1] = np.convolve(images[k][:,1], SOBEL)[1:-1]
    images[k][:,1] /= 4
    images[k] = pca.inverse_transform(images[k])

if FOLDER != "strips":
    missing = []

    for i in range(len(build)-1, 0, -1):
        idx1 = int(build[i][6:-4])
        idx2 = int(build[i-1][6:-4])
        count = abs(idx1 - idx2)
        if count > 1:
            build.insert(i, "WHITE")
            missing.append(i)
            pass


    h, channels = images[START_POINT].shape

    images["WHITE"] = np.full((h, 3), 255)

# Actually concatenate the image data together
built_image = np.concatenate([np.expand_dims(images[k],1).astype(np.uint8) for k in build], axis=1)
imageio.imwrite("export.png", built_image.astype(np.uint8))
logger.info("Image exported to ./export.png")
if FOLDER != "strips":
    for x, s in enumerate(scores):
        if(x in missing):
            plt.plot(x, s*10, marker='.', color="red")
        else:
            plt.plot(x, s*10, marker='.', color="black")
else:
    for x, s in enumerate(scores):
        if s*10 < 600:
            plt.plot(x, 1000-s, marker=',', color="red")
    plt.plot(score_edges)
plt.imshow(built_image)
plt.show()
