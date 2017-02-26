# Learning-to-navigate-without-a-map

This branch is a replicate version of Karpathy's blog.
It's not supposed to be merged with master branch.
Just for study purpose.

## Notes on OpenAI's Gym

Assume you are using Anaconda as your Python distribution like me.
And assume you want to install all systems.

### Anaconda

Install PyOpenGL by

```
conda install pyopengl
```

### Clone the source

```
git clone https://github.com/openai/gym.git
cd gym
```

### Mac OS X

Install these as following:

```
brew install cmake boost boost-python sdl2 swig wget
```

Run following to install everything:

```
export MACOSX_DEPLOYMENT_TARGET=10.11
pip install -e '.[all]'
```

### Ubuntu

Install the following

```
apt-get install -y cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev libboost-all-dev libsdl2-dev swig
```

Update `libgcc` in Anaconda

```
conda install libgcc
```

Install everything:

```
pip install -e '.[all]'
```


## Contacts

Yuhuang Hu  
Email: duguyue100@gmail.com
