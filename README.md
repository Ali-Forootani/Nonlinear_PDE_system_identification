# DeepyMod Excavation

DeePyMoD is a modular framework for model discovery of PDEs and ODEs from noise data. The framework is comprised of four components, that can seperately be altered: i) A function approximator to construct a surrogate of the data, ii) a function to construct the library of features, iii) a sparse regression algorithm to select the active components from the feature library and iv) a constraint on the function approximator, based on the active components.

In this repository I am trying to summarize the DeepyMod features step by step so that the reader can make development easily.

`Even though the authors mentioned the applicability of their package for the case of ODEs, one should face many difficulties to exploit this package for such cases. The reason is that data set preparation, custom library, and etc. have to be prepared from scratch. Therefore, I suggest to use it for the case of PDEs!`

## Downloading the package and exploring it

You can find the original DeepyMod package [here](https:https://github.com/PhIMaL/DeePyMoD).

Well if you extract it you will see 4 folders: (i) docs (ii) src (iii) tests (iv) examples.

In the folder `examples` you will find different jupyter notebook files(`.ipynb`) which are different `PDEs`.

These are the minimum requirements that you need for the simulation! Even though you can use, the following syntax
```bash
pip install deepymod
``` 
I recommend you to avoid such decision! because it throws bugs as you want to utilize the package!


## How to use and develop?

For doing this:

(i) Go to `src` folder and make `.py` file! or even `.ipynb` with any IDE (spyder or VSCode) or Jupyter notebook


then inside the empty `.py` file try to add the system path and import general libraries, such as:

```python
# General imports
import numpy as np
import torch
import sys
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
cwd = os.getcwd()
sys.path.append(cwd)
```
Then you can import modules that are required for our simulations:

```python
# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.data.burgers import burgers_delta
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold, STRidge
from deepymod.training import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
```

as you can see `deepymod` is the parent directory and the rest are subdirectories or modules. 
