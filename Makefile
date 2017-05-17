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

restructure-data-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/restructure_dataset.py

sample-train-ds:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/sample_train_ds.py

load-train-ds:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/load_train_ds.py

dstar-8-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dstar_8_exp.py

dstar-16-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dstar_16_exp.py

dstar-28-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dstar_28_exp.py

dstar-40-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dstar_40_exp.py

pg-16-exp:
	THEANO_FLAGS=device=cpu PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/pg_16_exp.py

ddpg-16-exp:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/ddpg_16_exp.py

world-generator:
	PYTHONPATH=$(PYTHONPATH) python ./world_generator/scripts/world_generator.py

# for 16x16
vin-exp-16:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/vin_exp_16.py

vin-exp-28:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/vin_exp_28.py

vin-exp-8:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/vin_exp_8.py

vin-exp-po-16:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/vin_exp_po_16.py

vin-exp-po-28:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/vin_exp_po_28.py

vin-exp-po-8:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/vin_exp_po_8.py

new-data-test:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/new_data_test.py

vis-vin-data:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vis_vin_data.py

vin-po-predict:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_po_predict.py

vin-predict:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_predict.py

vin-po-benchmark-export-16:
	THEANO_FLAGS=device=cpu PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_po_benchmark_export_16.py

vin-po-benchmark-export-28:
	THEANO_FLAGS=device=cpu PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_po_benchmark_export_28.py

vin-po-benchmark-export-8:
	THEANO_FLAGS=device=cpu PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_po_benchmark_export_8.py

vin-pg-16-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/vin_pg_16_exp.py

cleanall:
