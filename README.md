# [Learning to navigate without a map](http://dgyblog.com/projects-term/rlvision.html)
The goal of the project is to determine how well deep learning is suited for planning under incomplete information.

## Access data in HDF5 format

The Matlab format data is simply too slow to load.
Therefore I created some APIs to convert the raw data to
the HDF5 format and made them available in Google Drive

### Instructions to use the dataset

Before doing everything, make sure you've installed the package `h5py`,
for Anaconda user, you can simply do:

```
conda install h5py
```

1. Check you have the project resource folder at

```
$HOME/.rlvision
```

2. Copy HDF5 folder to following path

```
$HOME/.rlvision/data/
```

Note that `HDF5` folder should be a sub-folder of `data` folder

3. Check out the script `hdf5_data_test.py` in `tests` folder.
To run the script, simply type

```
make hdf5-data-test
```

You will see some lines of messages printed in the console,
and finally a plot for one grid.

4. In case you want to run the dstar algorithm with the dataset checkout `dstar_test.py` in `tests` folder
First, you will need to build the c++ source code, for that do the following:
* Install zeromq (\ØMQ\:): (this is only for ubuntu users, for others, please refer to (zeromq)[http://zeromq.org/intro:get-the-software])
```
sudo apt-get install libzmq3-dev
```
(**NOTE**: Currently the CMakeLists.txt assumes that libzmq is at /usr/local/lib, if you have issues while linking please refer to google: search for things like "link zmq library" zmq stands for (zeromq)[http://zeromq.org/]
* Move to the dstar-lite directory
* Create a build directory:
```
mkdir build
```
* Run cmake:
```
cmake ..
```
* Compile:
```
make
```
* Now you can return to the root directory of this repo where the Makefile is.
```
cd ../..
```

Second, to run the dstar example script, simpy type
```
make dstar_test.py
```
