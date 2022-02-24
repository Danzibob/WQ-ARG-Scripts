# WQ Un-Shredder

This is a repository of scripts to help solve the Destiny 2 Witch Queen ARG. Currently it can pull down TJ09's strips zip, extract them, and try to put the image together.

## Setup

To install the required python libraries using pip, run:

```bash
pip install -r requirements.txt
```

If you wish to run the shell scripts provided in the `justfile`, you also need to install [Just](https://github.com/casey/just). Alternatively, you can manually run the commands in this file, or download and unzip the strips folder yourself.

## Usage

Using the `justfile`, running the program is as simple as:
```bash
just build-image
```
This will pull down the latest strips zip, extract it into a folder called `strips`, and run unshred.py. You can of course run all these steps manually.

Also included in the repo is a folder called `sample_pages`. These are images taken from the scanned in Collector's Edition book, that we're using as training data. To shred these images and generate a set of strips from them, run `just shred-pages`. This will create a folder called `sample_strips`. Changing the `FOLDER` variable at the top of `unshred.py` to `sample_strips` will then run the gluing code on this sample data, to give us an idea of how it will perform once we have all of the data.

### A note on cache
The program caches similarity scores between strips, so it doesn't have to recalculate them all every time you run it. As such, if you change the similarity function you _must also delete the similarity cache_. If you don't do this, your altered similarity function will continue to use the old cached values from before the code change.