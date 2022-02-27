get-images:
	wget -O clustered2.json https://tjl.co/wqarg/clustered2.json
	wget -O mapping.json https://tjl.co/wqarg/mapping.json
	wget -O strips.tar.gz https://tjl.co/wqarg/strips.tar.gz
	
	tar -xvf strips.tar.gz > /dev/null
	ls strips | wc -l

shred-pages:
	mkdir -p sample_strips
	python gen_training_strips.py

build-image: get-images
	python unshredV2.py -c clustered2.json -i strips -o full_export.png
