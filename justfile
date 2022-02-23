get-images:
	wget -O strips.tar.gz https://tjl.co/wqarg/strips.tar.gz
	tar -xvf strips.tar.gz > /dev/null
	ls strips | wc -l

shred-pages:
	mkdir -p sample_strips
	python gen_training_strips.py

build-image: get-images
	python unshred.py
