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


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(50)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#########################


# Making dataset
v = 0.1
A = 1.0

x = torch.linspace(-3, 4, 100)
t = torch.linspace(0.5, 5.0, 50)


#x = torch.tensor(x)
#t = torch.tensor(t)


load_kwargs = {"x": x, "t": t, "v": v, "A": A}
preprocess_kwargs = {"noise_level": 0.00}


#########################
#########################
#########################

dataset = Dataset(
    burgers_delta,
    load_kwargs=load_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": 500},
    device=device,
)

coords = dataset.get_coords().cpu()
data = dataset.get_data().cpu()
fig, ax = plt.subplots()
im = ax.scatter(coords[:,1], coords[:,0], c=data[:,0], marker="x", s=10)
ax.set_xlabel('x')
ax.set_ylabel('t')
fig.colorbar(mappable=im)

plt.show()


##########################
##########################


train_dataloader, test_dataloader = get_train_test_loader(
    dataset, train_test_split=0.8)


##########################
##########################

poly_order = 2
diff_order = 2

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1


network = NN(2, [64, 64, 64, 64], 1)

library = Library1D(poly_order, diff_order)
estimator = Threshold(0.1) 
sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5)
constraint = LeastSquares()
constraint_2 = Ridge()
constraint_3 = STRidgeCons()

estimator_2 = STRidge()

#linear_module = CoeffsNetwork(int(n_combinations),int(n_features))


#constraint = Ridge()
# Configuration of the sparsity scheduler
model = DeepMoD(network, library, estimator_2, constraint_3, estimator_2).to(device)


# Defining optimizer
optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3) 



train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    exp_ID="Test",
    write_iterations=25,
    max_iterations=25000,
    delta=1e-4,
    patience=200,
)

model.sparsity_masks

print(model.estimator_coeffs())
print(model.constraint.coeff_vectors[0].detach().cpu())

```

## DeepMOD class


Ok, let's have a close look at `DeepMoD`. It is inherited from `nn.Module` well you need mainly to work with its `__.init__` and `forward` methods. 


```python

class DeepMoD(nn.Module):
    def __init__(
        self,
        function_approximator: torch.nn.Sequential,
        library: Library,
        sparsity_estimator: Estimator,
        constraint: Constraint,
        sparsity_estimator_2: Estimator
    ) -> None:
        """The DeepMoD class integrates the various buiding blocks into one module. The function approximator approximates the data,
        the library than builds a feature matrix from its output and the constraint constrains these. The sparsity estimator is called
        during training to update the sparsity mask (i.e. which terms the constraint is allowed to use.)

        Args:
            function_approximator (torch.nn.Sequential): [description]
            library (Library): [description]
            sparsity_estimator (Estimator): [description]
            constraint (Constraint): [description]
        """
        super().__init__()
        self.func_approx = function_approximator
        self.library = library
        self.sparse_estimator = sparsity_estimator
        self.constraint = constraint
        self.sparse_estimator_2 = sparsity_estimator_2

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, TensorList, TensorList]:
        """The forward pass approximates the data, builds the time derivative and feature matrices
        and applies the constraint.
        It returns the prediction of the network, the time derivatives and the feature matrices.

        Args:
            input (torch.Tensor):  Tensor of shape (n_samples, n_outputs) containing the coordinates, first column should be the time coordinate.

        Returns:
            Tuple[torch.Tensor, TensorList, TensorList]: The prediction, time derivatives and and feature matrices of respective sizes                                                       ((n_samples, n_outputs), [(n_samples, 1) x n_outputs]), [(n_samples, n_features) x n_outputs])
        """
        prediction, coordinates = self.func_approx(input)
        time_derivs, thetas = self.library((prediction, coordinates))
        
        input_pairs = (time_derivs, thetas)
        coeff_vectors = self.constraint(time_derivs, thetas)
        
        return prediction, time_derivs, thetas
```

now we go into the details how to make an instance from DeepMOD class:

`function_approximator:` nn.module object

```python
network = NN(2, [64, 64, 64, 64], 1)
```
constructing a MLP module, you can find its root here:

```python
from deepymod.model.func_approx import NN
```
the default activation function of this MLP is SIREN activation function you can find it [here](https://github.com/vsitzmann/siren)

Moreover, as you know we deal with spacious-temporal data sets therefore the input as well as outputs of our MLP should be considered based on the type of the PDE that we deal with. e.g. here it has `2` inputs, i.e. `(t,x)`, time and state and only `1` output.

to make an instance `library` module see the following
```python
from deepymod.model.library import Library1D
library = Library1D(poly_order, diff_order)
```
well library module has 2 classes such as `Library2D` or Library1D and 2 functions.

the `Library1D` inherits from the following base class in `deepmod` module: 

```python
class Library(nn.Module):
    def __init__(self) -> None:
        """Abstract baseclass for the library module."""
        super().__init__()
        self.norms = None

    def forward(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[TensorList, TensorList]:
        """Compute the library (time derivatives and thetas) from a given dataset. Also calculates the norms
        of these, later used to calculate the normalized coefficients.

        Args:
            input (Tuple[TensorList, TensorList]): (prediction, data) tuple of size ((n_samples, n_outputs), (n_samples, n_dims))

        Returns:
            Tuple[TensorList, TensorList]: Temporal derivative and libraries of size ([(n_samples, 1) x n_outputs]), [(n_samples, n_features)x n_outputs])
        """
        time_derivs, thetas = self.library(input)
        self.norms = [
            (torch.norm(time_deriv) / torch.norm(theta, dim=0, keepdim=True))
            .detach()
            .squeeze()
            for time_deriv, theta in zip(time_derivs, thetas)
        ]
        return time_derivs, thetas

    @abstractmethod
    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[TensorList, TensorList]:
        """Abstract method. Specific method should calculate the temporal derivative and feature matrices.
        These should be a list; one temporal derivative and feature matrix per output.

        Args:
        input (Tuple[TensorList, TensorList]): (prediction, data) tuple of size ((n_samples, n_outputs), (n_samples, n_dims))

        Returns:
        Tuple[TensorList, TensorList]: Temporal derivative and libraries of size ([(n_samples, 1) x n_outputs]), [(n_samples, n_features)x n_outputs])
        """
        pass
```

as we can see it has an abstract method, therefore any class (Library1D/Library2D) that inherits from it should have a `library` method in itself.

This means that a custom library class can be made by re-writing its `def library()` method.



Now we need to define the method for sparsity and computing the coefficients:

```python
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from deepymod.model.sparse_estimators import Threshold, STRidge
```

I modified STRidge from this package [PDEFIND](https://github.com/snagcliffs/PDE-FIND) and made it as a new class and compatible for our framework.


Both base Estimator and Constraint have abstract methods, `def fit()`,


### Constraint Base class

```python

class Constraint(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the constraint module."""
        super().__init__()
        self.sparsity_masks: TensorList = None

    def forward(self, time_derivs, thetas) -> TensorList:
        
        #def forward(self, input: Tuple[TensorList, TensorList]) -> TensorList:

        
        """The forward pass of the constraint module applies the sparsity mask to the
        feature matrix theta, and then calculates the coefficients according to the
        method in the child.

        Args:
            input (Tuple[TensorList, TensorList]): (time_derivs, library) tuple of size
                    ([(n_samples, 1) X n_outputs], [(n_samples, n_features) x n_outputs]).
        Returns:
            coeff_vectors (TensorList): List with coefficient vectors of size ([(n_features, 1) x n_outputs])
        """
        
        #time_derivs, thetas = input
        
        if self.sparsity_masks is None:
            self.sparsity_masks = [
                torch.ones(theta.shape[1], dtype=torch.bool).to(theta.device)
                for theta in thetas
            ]

        sparse_thetas = self.apply_mask(thetas, self.sparsity_masks)

        # Constraint grad. desc style doesn't allow to change shape, so we return full coeff
        # and multiply by mask to set zeros. For least squares-style, we need to put in
        # zeros in the right spot to get correct shape.
        
        
        """
        with torch.no_grad():
            coeff_vectors = [ self.fit(sparse_thetas[0].detach().cpu().numpy(),
                                     time_derivs[0].detach().cpu().numpy()).to(thetas[0].device)
                             ]
        """    
        coeff_vectors =  self.fit(sparse_thetas, time_derivs)
        
        #coeff_vectors = self.fit(sparse_thetas, time_derivs)
        
        
        self.coeff_vectors = [
            self.map_coeffs(mask, coeff)
            if mask.shape[0] != coeff.shape[0]
            else coeff * mask[:, None]
            for mask, coeff in zip(self.sparsity_masks, coeff_vectors)
        ]
        
       
        
        return self.coeff_vectors

    @staticmethod
    def apply_mask(thetas: TensorList, masks: TensorList) -> TensorList:
        """Applies the sparsity mask to the feature (library) matrix.

        Args:
            thetas (TensorList): List of all library matrices of size [(n_samples, n_features) x n_outputs].

        Returns:
            TensorList: The sparse version of the library matrices of size [(n_samples, n_active_features) x n_outputs].
        """
        sparse_thetas = [theta[:, mask] for theta, mask in zip(thetas, masks)]
        return sparse_thetas

    @staticmethod
    def map_coeffs(mask: torch.Tensor, coeff_vector: torch.Tensor) -> torch.Tensor:
        """Places the coeff_vector components in the true positions of the mask.
        I.e. maps ((0, 1, 1, 0), (0.5, 1.5)) -> (0, 0.5, 1.5, 0).

        Args:
            mask (torch.Tensor): Boolean mask describing active components.
            coeff_vector (torch.Tensor): Vector with active-components.

        Returns:
            mapped_coeffs (torch.Tensor): mapped coefficients.
        """
        mapped_coeffs = (
            torch.zeros((mask.shape[0], 1))
            .to(coeff_vector.device)
            .masked_scatter_(mask[:, None], coeff_vector)
        )
        return mapped_coeffs

    @abstractmethod
    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Abstract method. Specific method should return the coefficients as calculated from the sparse feature
        matrices and temporal derivatives.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): Calculated coefficients of size (n_active_features, n_outputs).
        """
        raise NotImplementedError
```
####################################
####################################


### Estimator Base class

```python

class Estimator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the sparse estimator module."""
        super().__init__()
        self.coeff_vectors = None

    def forward(self, thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """The forward pass of the sparse estimator module first normalizes the library matrices
        and time derivatives by dividing each column (i.e. feature) by their l2 norm, than calculate the coefficient vectors
        according to the sparse estimation algorithm supplied by the child and finally returns the sparsity
        mask (i.e. which terms are active) based on these coefficients.

        Args:
            thetas (TensorList): List containing the sparse feature tensors of size  [(n_samples, n_active_features) x n_outputs].
            time_derivs (TensorList): List containing the time derivatives of size  [(n_samples, 1) x n_outputs].

        Returns:
            (TensorList): List containting the sparsity masks of a boolean type and size  [(n_samples, n_features) x n_outputs].
        """

        # we first normalize theta and the time deriv
        with torch.no_grad():
            normed_time_derivs = [(time_derivs[0] / torch.norm(time_derivs[0])).detach().cpu().numpy()]
            
            [
                (time_deriv / torch.norm(time_deriv)).detach().cpu().numpy()
                for time_deriv in time_derivs
            ]
            normed_thetas = [(thetas[0] / torch.norm(thetas[0], dim=0, keepdim=True)).detach().cpu().numpy()]
            
            [
                (theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu().numpy()
                for theta in thetas
            ]

        self.coeff_vectors = [
            self.fit(theta, time_deriv.squeeze())[:, None]
            for theta, time_deriv in zip(normed_thetas, normed_time_derivs)
        ]
        sparsity_masks = [
            torch.tensor(coeff_vector != 0.0, dtype=torch.bool)
            .squeeze()
            .to(thetas[0].device)  # move to gpu if required
            for coeff_vector in self.coeff_vectors
        ]
        
        #print(sparsity_masks)
        
        return sparsity_masks

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Abstract method. Specific method should compute the coefficient based on feature matrix X and observations y.
        Note that we expect X and y to be numpy arrays, i.e. this module is non-differentiable.

        Args:
            x (np.ndarray): Feature matrix of size (n_samples, n_features)
            y (np.ndarray): observations of size (n_samples, n_outputs)

        Returns:
            (np.ndarray): Coefficients of size (n_samples, n_outputs)
        """
        pass
```

Let's have a close look at `Estimator` base class. We first divide the dictionary terms and time derivatives by their 2-norms.  

In DeepMOD they considered to put everything within a list. 

so if you see here I simplified to the following form
```python
with torch.no_grad():

      normed_time_derivs = [(time_derivs[0] / torch.norm(time_derivs[0])).detach().cpu().numpy()]

      normed_thetas = [(thetas[0] / torch.norm(thetas[0], dim=0, keepdim=True)).detach().cpu().numpy()]            
```

The main reason to use `with torch.no_grad():` is that we avoid any change in the training of deep learning. 

Moreover, the above syn-taxes can be written alternatively as follows:
```python
with torch.no_grad():


            [
                (time_deriv / torch.norm(time_deriv)).detach().cpu().numpy()
                for time_deriv in time_derivs
            ]
            normed_thetas = [(thetas[0] / torch.norm(thetas[0], dim=0, keepdim=True)).detach().cpu().numpy()]
            
            [
                (theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu().numpy()
                for theta in thetas
            ]
```
so we can remove the redundant parts accordingly.

After this step, we can feed the normalized versions to a least square framework by the following syntax:

```python
self.coeff_vectors = [
            self.fit(theta, time_deriv.squeeze())[:, None]
            for theta, time_deriv in zip(normed_thetas, normed_time_derivs)
        ]
```

well the same as before we can make it more simple! anyway, we leave it as it is.

finally we can make a `sparsity_masks` as follows:

```python
sparsity_masks = [
            torch.tensor(coeff_vector != 0.0, dtype=torch.bool)
            .squeeze()
            .to(thetas[0].device)  # move to gpu if required
            for coeff_vector in self.coeff_vectors
        ]
```

consider the term `torch.tensor(coeff_vector != 0.0, dtype=torch.bool)`! which make a boolean tensor corresponding to non-zero elements of a torch.tensor!



```python

class Estimator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the sparse estimator module."""
        super().__init__()
        self.coeff_vectors = None

    def forward(self, thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """The forward pass of the sparse estimator module first normalizes the library matrices
        and time derivatives by dividing each column (i.e. feature) by their l2 norm, than calculate the coefficient vectors
        according to the sparse estimation algorithm supplied by the child and finally returns the sparsity
        mask (i.e. which terms are active) based on these coefficients.

        Args:
            thetas (TensorList): List containing the sparse feature tensors of size  [(n_samples, n_active_features) x n_outputs].
            time_derivs (TensorList): List containing the time derivatives of size  [(n_samples, 1) x n_outputs].

        Returns:
            (TensorList): List containting the sparsity masks of a boolean type and size  [(n_samples, n_features) x n_outputs].
        """

        # we first normalize theta and the time deriv
        with torch.no_grad():
            normed_time_derivs = [(time_derivs[0] / torch.norm(time_derivs[0])).detach().cpu().numpy()]
            
            [
                (time_deriv / torch.norm(time_deriv)).detach().cpu().numpy()
                for time_deriv in time_derivs
            ]
            normed_thetas = [(thetas[0] / torch.norm(thetas[0], dim=0, keepdim=True)).detach().cpu().numpy()]
            
            [
                (theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu().numpy()
                for theta in thetas
            ]

        self.coeff_vectors = [
            self.fit(theta, time_deriv.squeeze())[:, None]
            for theta, time_deriv in zip(normed_thetas, normed_time_derivs)
        ]
        sparsity_masks = [
            torch.tensor(coeff_vector != 0.0, dtype=torch.bool)
            .squeeze()
            .to(thetas[0].device)  # move to gpu if required
            for coeff_vector in self.coeff_vectors
        ]
        
        #print(sparsity_masks)
        
        return sparsity_masks

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Abstract method. Specific method should compute the coefficient based on feature matrix X and observations y.
        Note that we expect X and y to be numpy arrays, i.e. this module is non-differentiable.

        Args:
            x (np.ndarray): Feature matrix of size (n_samples, n_features)
            y (np.ndarray): observations of size (n_samples, n_outputs)

        Returns:
            (np.ndarray): Coefficients of size (n_samples, n_outputs)
        """
        pass
```

#############################################

by taking a close look at the `Constraint` class we see that instance attribute `self.sparsity_masks` gets its value from forward method of `Estimator` class! Just consider the return value of `Estimator.forward` which is `sparsity_masks`. As a matter of the default value of `saprsity_masks` in `Constraint` class is `None`, however during the training loop it is initialized by the following syntax:

```python
model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
```

Let's go into the details:

```python
 if self.sparsity_masks is None:
            self.sparsity_masks = [
                torch.ones(theta.shape[1], dtype=torch.bool).to(theta.device)
                for theta in thetas
            ]

        sparse_thetas = self.apply_mask(thetas, self.sparsity_masks)
```
the `if self.sparsity_masks is None: ...` part is obvious. The next calls a method inside the `Constraint` class, which is defined as static method as follows:

```python
@staticmethod
    def apply_mask(thetas: TensorList, masks: TensorList) -> TensorList:
        """Applies the sparsity mask to the feature (library) matrix.

        Args:
            thetas (TensorList): List of all library matrices of size [(n_samples, n_features) x n_outputs].

        Returns:
            TensorList: The sparse version of the library matrices of size [(n_samples, n_active_features) x n_outputs].
        """
        sparse_thetas = [theta[:, mask] for theta, mask in zip(thetas, masks)]
        return sparse_thetas
```
the line `[theta[:, mask] for theta, mask in zip(thetas, masks)]` just exclude columns in dictionary that are `False` in `sparsity_masks` (or mask).


then we feed `sparse_thetas` and `time derivative` to another `self.fit` 

```python
coeff_vectors =  self.fit(sparse_thetas, time_derivs)
```
like before we can use many approaches to solve this least square problem as follows:

```python
from deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
```
anyway. `self.fit` is an abstract method and any custom library class has to implement `def fit()` method! this is a typical in design pattern approaches. the results of this `coeff_vectors` has a size equal to the size of `non-False` elements of `sparsity_masks` therefore we should bring them back into the original shape! thus we make use of the following:

```python
self.coeff_vectors = [
            self.map_coeffs(mask, coeff)
            if mask.shape[0] != coeff.shape[0]
            else coeff * mask[:, None]
            for mask, coeff in zip(self.sparsity_masks, coeff_vectors)
        ]
@staticmethod
    def map_coeffs(mask: torch.Tensor, coeff_vector: torch.Tensor) -> torch.Tensor:
        """Places the coeff_vector components in the true positions of the mask.
        I.e. maps ((0, 1, 1, 0), (0.5, 1.5)) -> (0, 0.5, 1.5, 0).

        Args:
            mask (torch.Tensor): Boolean mask describing active components.
            coeff_vector (torch.Tensor): Vector with active-components.

        Returns:
            mapped_coeffs (torch.Tensor): mapped coefficients.
        """
        mapped_coeffs = (
            torch.zeros((mask.shape[0], 1))
            .to(coeff_vector.device)
            .masked_scatter_(mask[:, None], coeff_vector)
        )
        return mapped_coeffs
```

the last step is to 



############################################

### DeepMOD.forward


let's consider again `forward` method of DeepMOD base class. We feed our data set to the NN module, then we get the prediction and the coordinates. However, the coordinates are the dame as the data set! 

We feed the prediction to the library module as a tuple to compute the time derivative and the dictionary!!!


then we need to solve the `time derivative = Dictionary * coeffs`. To solve this equation, we already defined `LeastSquares, Ridge, STRidgeCons` approaches. So we can feed them as well.

```python
def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, TensorList, TensorList]:
        """The forward pass approximates the data, builds the time derivative and feature matrices
        and applies the constraint.

        It returns the prediction of the network, the time derivatives and the feature matrices.

        Args:
            input (torch.Tensor):  Tensor of shape (n_samples, n_outputs) containing the coordinates, first column should be the time coordinate.

        Returns:
            Tuple[torch.Tensor, TensorList, TensorList]: The prediction, time derivatives and and feature matrices of respective sizes
                                                       ((n_samples, n_outputs), [(n_samples, 1) x n_outputs]), [(n_samples, n_features) x n_outputs])

        """
        prediction, coordinates = self.func_approx(input)
        time_derivs, thetas = self.library((prediction, coordinates))
        
        input_pairs = (time_derivs, thetas)
        
        coeff_vectors = self.constraint(time_derivs, thetas)
        
        
        return prediction, time_derivs, thetas
``` 

##############################################

Let's set up our framework for training! We need to make an instance from different base class! order of the library, estimator method, constraint, and etc.

There are some redundancy as well in DeepMOD such as `rainTestPeriodic` base class. To tell truth we do not need it in the training loop and simply we can comment it! But make the comment in the training loop not anywhere else!


```python
poly_order = 2
diff_order = 2
n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1

network = NN(2, [64, 64, 64, 64], 1)
library = Library1D(poly_order, diff_order)
estimator = Threshold(0.1) 
sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5)
constraint = LeastSquares()
constraint_2 = Ridge()
constraint_3 = STRidgeCons()
estimator_2 = STRidge()


model = DeepMoD(network, library, estimator_2, constraint_3, estimator_2).to(device)
```











```python
    @property
    def sparsity_masks(self):
        """Returns the sparsity masks which contain the active terms."""
        return self.constraint.sparsity_masks

    def estimator_coeffs(self) -> TensorList:
        """Calculate the coefficients as estimated by the sparse estimator.

        Returns:
            (TensorList): List of coefficients of size [(n_features, 1) x n_outputs]
        """
        coeff_vectors = self.constraint.coeff_vectors
        return coeff_vectors

    def constraint_coeffs(self, scaled=False, sparse=False) -> TensorList:
        """Calculate the coefficients as estimated by the constraint.

        Args:
            scaled (bool): Determine whether or not the coefficients should be normalized
            sparse (bool): Whether to apply the sparsity mask to the coefficients.

        Returns:
            (TensorList): List of coefficients of size [(n_features, 1) x n_outputs]
        """
        coeff_vectors = self.constraint.coeff_vectors
        if scaled:
            coeff_vectors = [
                coeff / norm[:, None]
                for coeff, norm, mask in zip(
                    coeff_vectors, self.library.norms, self.sparsity_masks
                )
            ]
        if sparse:
            coeff_vectors = [
                sparsity_mask[:, None] * coeff
                for sparsity_mask, coeff in zip(self.sparsity_masks, coeff_vectors)
            ]
        return coeff_vectors
```




## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
