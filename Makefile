# This is a Python template Makefile, do modification as you want
#
# Project: 
# Author:
# Email :

HOST = 127.0.0.1
PYTHONPATH="$(shell printenv PYTHONPATH):$(PWD)"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

run:

test:
	PYTHONPATH=$(PYTHONPATH) python 

data-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/data_test.py

load-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/data_load_test.py

hdf5-data-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/hdf5_data_test.py

grid-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/grid_test.py

dstar-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/dstar_test.py

grid-class-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/grid_class_test.py




cleanall:
