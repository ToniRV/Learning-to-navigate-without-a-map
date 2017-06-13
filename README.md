# [Learning to navigate without a map](http://dgyblog.com/projects-term/rlvision.html)
The goal of the project is to determine how well deep learning is suited for planning under incomplete information.


## Instructions

### Requirements

+ Python 2.7 (In principle, this project can be run under Python 3, we didn't test it however)
+ Keras 2
+ Some specific packages in `requirements.txt`

### D-star package compilation

Please follow the instruction in [Dstar implementation](./dstar-lite)

### How to use this package

1. Clone this package:

```
$ git clone https://github.com/ToniRV/Learning-to-navigate-without-a-map
```

2. Check you have the project resource folder at

```
$HOME/.rlvision
```

__Note that this folder will be automatically created at the first time that
you run the package, you can get the correct resource folder by__ 

```
$ python ./rlvision/__init__.py
```

3. Copy data to the `data` folder

## Contacts

Yuhuang Hu, Shu Liu, Antoni Rosiñol Vidal, Yang Yu  
Email: {hyh, liush, antonir, yuya}@student.ethz.ch
