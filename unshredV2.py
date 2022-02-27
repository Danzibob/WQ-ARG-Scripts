import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--regen", help="Regenerate the json cache file from scratch", action="store_true")
    parser.add_argument("-x", "--nocombine", help="Halt the program after calculating similarities", action="store_true")
    parser.add_argument("-q", "--quiet", help="Don't open any plots", action="store_true")
    parser.add_argument("-K", "--keysonly", help="Halt program after outputting keys (i.e. don't export images)", action="store_true")

    parser.add_argument("-i", "--input", help="Path of the strips folder", default="strips")
    parser.add_argument("-o", "--output", help="Path of the output image folder", default="export.png")
    parser.add_argument("-m", "--mapping", help="Path of the name mapping json", default="mapping.json")
    parser.add_argument("-f", "--cachefile", help="Path of the json cache file", default="strips-similarity_cache.json")
    parser.add_argument("-c", "--clusters", help="Specify a set of clusters on which the program will run seperately", default="clustered2.json")
    parser.add_argument("-k", "--keylist", help="Output the constructed list of image keys to the specified file", default="built_keys.json")
    parser.add_argument("-t", "--threads", help="Number of threads to use for muti-processing", type=int, default=8)
    args = parser.parse_args()
    return args

parse_args()

from functools import lru_cache
import imageio, os, json, sys
import numpy as np
import logging
from multiprocess import Pool, current_process
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from random import randint, shuffle, choice
from time import time
from tqdm import tqdm

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
MINUS_INFINITY = float('-inf')

logging.basicConfig(level=logging.WARNING, format='%(levelname)s | %(message)s')
lg = logging.getLogger(__name__)

lg.setLevel(logging.INFO)

# Load a single image into a numpy array
def read_img(path):
    im = imageio.imread(path)
    # Remove alpha channel if present
    if im.shape[2] == 4:
        im = np.delete(im, [3], 2)
    return im.squeeze()

# Finds the paths and short-keys for all images in the input folder
def collect_image_paths(folder, mapping):
    paths = {}
    for f in os.listdir(folder):
        try:
            key = mapping[f[:-6]] + f[-6:-4]
        except KeyError:
            lg.warn(f"File '{f}' didn't have a name mapping listed in the provided file")
        paths[key] = os.path.join(folder, f)
    lg.info(f"Found {len(paths)} files in ./{folder}")
    return paths

# Builds a matrix of pre-computed colour distances
def compute_colour_distances(palette):
    # Pass vals through a log function so non-black colours are closer together
    # scaled = -10000 / (palette+100) + 100
    scaled = palette.astype(np.float64)
    # But also make the blue channel farther apart than the others
    scaled[:,2] *= 1.5

    matrix = {}

    for i, c2 in enumerate(palette):
        for j, c1 in enumerate(palette):
            if j > i: break
            v = np.linalg.norm(c1-c2)
            matrix[PRIMES[i] * PRIMES[j]] = v
    return matrix

# Expresses the similarity between strips
OFFSETS = (-1,0,1)
WEIGHTS = (0.2,1,0.2)
def similarity(a,b, matrix):
    # ---=== THE STRIP SIMILARITY ALGORITHM ===---
    # This section returns a similarity score between the two provided strips
    #
    # The algorithm provided looks at the adjacent pixel, as well as pixels
    # One above and one below that pixel, then sums 
    # 
    # NB: This function needs to be pretty heckin fast to not be a bottleneck
    score = 0
    for roll, weight in zip(OFFSETS,WEIGHTS):
        score += sum((np.roll(matrix[x], roll) for x in a*b))
    return 100/(score/len(a)/1.4 + 10)



if __name__ == "__main__":
    args = parse_args()

    # Load mappings json
    try:
        with open(args.mapping) as f:
            mapping = json.load(f)
    except Exception as e:
        lg.error("Fatal - failed to parse mappings file.")
        raise e
    lg.info("Loaded mappings")

    # Load clusters json
    try:
        with open(args.clusters) as f:
            clusters_json = json.load(f)
    except Exception as e:
        lg.error("Fatal - failed to parse clusters file.")
        raise e
    lg.info("Loaded clusters")

    # Load cache
    if not args.regen and os.path.exists(args.cachefile):
        lg.info("Found existing similarity cache, loading...")
        with open(args.cachefile) as f:
            sims = json.load(f)
    else:
        lg.info("No similarity cache found, building from scratch...")
        sims = {}

    # Parse input image folder
    files = collect_image_paths(args.input, mapping)
    
    # Select images to use
    clusters = []
    for cluster in clusters_json:
        cluster_keys = []
        for s in cluster:
            key = mapping[s[:-2]] + s[-2:]
            if key in files:
                cluster_keys.append(key)
            else:
               lg.warn("Key {} wasn't present in the input folder ")
        clusters.append(cluster_keys)
    lg.info("Selected clustered strips")
    
    # Load required images
    needed = set(np.concatenate(clusters))
    images = {}
    for key in needed:
        images[key] = read_img(files[key])
    lg.info(f"Loaded {len(needed)} image files")

    # Function to calculate strip similarities within a cluster
    def cluster_sims(cluster_num):
        cluster = clusters[cluster_num]
        threadid = current_process()._identity
        lg.info(f"Thread {threadid} is now working on cluster {cluster_num}")

        # Extract the colours for this cluster
        all_colours = np.concatenate([images[x] for x in cluster])
        palette = np.unique(all_colours, axis=0)

        lg.debug(all_colours)
        lg.debug(f"\tCluster {cluster_num}: Extracted {len(all_colours)} colours in")

        # Precompute distances between colours
        color_matrix = compute_colour_distances(palette)

        # Convert images to use palette indexes
        palette = [list(c) for c in palette]
        pal_imgs = {}
        for key in cluster:
            pal_imgs[key] = np.array([PRIMES[palette.index(list(c))] for c in images[key]])
        
        lg.debug(f"\tCluster {cluster_num}: Converted images to pallete indexes")
        
        # Calculate missing similarities
        cached_keys = sims.keys()
        new_sims = {}
        for i, k1 in enumerate(cluster):
            for k2 in cluster[:i+1]:
                sim_key = min(k1, k2) + "," + max(k1,k2)
                if sim_key not in cached_keys:
                    new_sims[sim_key] = similarity(pal_imgs[k1], pal_imgs[k2], color_matrix)
            if i % 100 == 0:
                lg.debug(f"\tCluster {cluster_num}: Calculating {i}/{len(cluster)} images")
        lg.info(f"\tCluster {cluster_num}: Done!")

        return new_sims

    # Calculate each of the clusters' similarities using multiple threads
    pool = Pool(args.threads)
    cluster_similarities = pool.map(cluster_sims, range(len(clusters)))
    
    # Collect similarity scores into a single dict
    for c_s in cluster_similarities:
        sims.update(c_s)
    
    lg.info(f"SIMS LENGTH: {len(sims)}")

    # Cache similarities to disk
    with open(args.cachefile, 'w') as outfile:
        json.dump(sims, outfile)

    lg.info(f"Saved cache at {args.cachefile}")

    if (args.nocombine):
        lg.info("No-combine flag was specified, halting program here.")
        pool.close()
        sys.exit()
    
    # Puts together clusters' strips using similarities calculated above
    def build_image(cluster_num):
        cluster = clusters[cluster_num]
        threadid = current_process()._identity
        lg.info(f"Thread {threadid} is now building cluster {cluster_num}")


        # ---=== THE STITCHING ALGORITHM ===---
        # Using the strip similarities calculated above
        # the stitching algorithm decides how clusters should be assembled
        # expected output is a list of short-form keys
        # (i.e.) an ordering of the "cluster" variable
        # 
        # The algorithm included here starts from one end and simply
        # chooses the best match for that strip glues them together
        # repeating until all strips are used.

        # Start from a single strip (minimum here to be consistent)
        START_POINT = min(cluster)
        build = [START_POINT]
        # remaining tracks the yet-to-be-stitched strips
        remaining = set(cluster)
        remaining.remove(START_POINT)
        # scores tracks the similairty between the strips we decided to stitch
        # Just so it's easier to plot later
        scores = []
        lg.debug(f"Starting with key {START_POINT}")

        # While there are remaining strips
        while len(remaining) > 0:
            best_score = MINUS_INFINITY
            best_idx = None
            # k1 is the key of the current last strip
            k1 = build[-1]
            # for each remaining strip
            for k2 in remaining:
                # find if the similarity is better than the current best
                sim_key = min(k1, k2) + "," + max(k1,k2)
                if sims[sim_key] > best_score:
                    # and update them if it is
                    best_score = sims[sim_key]
                    best_idx = k2

            lg.debug(f"{len(build)}) Best Score was {best_score} at index {best_idx}")
            # Then add that strip to the build
            build.append(best_idx)
            # And that score to the scores list
            scores.append(best_score)

            # And remove that strip from the remaining list
            remaining.remove(best_idx)

        lg.info("\tCluster {cluster_num}: Assembling Complete!")
        return (build, scores)

    built_clusters = pool.map(build_image, range(len(clusters)))
    builds, scores = zip(*built_clusters)
    print(builds[0])
    print(scores[0])

    with open(args.keylist, 'w') as outfile:
        json.dump(builds, outfile)
    lg.info(f"Cluster key orderings exported to {args.keylist}")
    
    if (args.keysonly):
        lg.info("Keys-only flag was specified, halting program here.")
        pool.close()
        sys.exit()

    # Make sure we close the thread pool
    pool.close()
   
    pages = sum(builds, [])
    page_scores = sum(scores, [])

    # Actually concatenate the image data together
    built_image = np.concatenate([np.expand_dims(images[k],1).astype(np.uint8) for k in pages], axis=1)
    imageio.imwrite(args.output, built_image.astype(np.uint8))
    lg.info(f"Image exported to {args.output}")
    if not args.quiet:
        for x, s in enumerate(page_scores):
            plt.plot(x, 1000-(s*1000), marker=',', color="red")
        # plt.plot(score_edges)
        plt.imshow(built_image)
        plt.show()
