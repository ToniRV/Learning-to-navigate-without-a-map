# This is a Python template Makefile, do modification as you want
#
# Project: Learning to navigate without a map 
# Author: Yuhuang Hu, Shu Liu, Antoni Rosiñol Vidal, Yang Yu
# Email : {hyh, liush, antonir, yuya}@student.ethz.ch

HOST = 127.0.0.1
PYTHONPATH="$(shell printenv PYTHONPATH):$(PWD)"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

run:

# Experiments

## Dstar
dstar-8-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dstar_8_exp.py

dstar-16-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dstar_16_exp.py

dstar-28-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dstar_28_exp.py


## DQN
dqn-8-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dqn_8_exp.py

dqn-16-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dqn_16_exp.py

dqn-28-exp:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/dqn_28_exp.py


## PG
pg-16-exp:
	THEANO_FLAGS=device=cpu PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/pg_16_exp.py

ddpg-16-exp:
	KERAS_BACKEND=tensorflow PYTHONPATH=$(PYTHONPATH) python ./rlvision/exps/ddpg_16_exp.py

## VIN
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

# Tests

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


world-generator:
	PYTHONPATH=$(PYTHONPATH) python ./world_generator/scripts/world_generator.py

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

vin-po-export-16:
	THEANO_FLAGS=device=cpu PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_po_export_value_reward_16.py

vin-po-export-28:
	THEANO_FLAGS=device=cpu PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_po_export_value_reward_28.py

vin-po-result:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/vin_po_result.py

select-po-grid:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/select_po_grid.py

select-po-grid-export:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/select_po_grid_export.py

select-dstar-grid-export:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/select_dstar_grid_export.py

load-selected-grid:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/load_selected_grid.py

plot-loss-fig:
	PYTHONPATH=$(PYTHONPATH) python ./rlvision/tests/plot_loss_fig.py


cleanall:
