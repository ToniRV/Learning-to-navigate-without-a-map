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

pong-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlpong/tests/ddpg_pong.py

keras-pong-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlpong/tests/keras_pong.py

cleanall:
