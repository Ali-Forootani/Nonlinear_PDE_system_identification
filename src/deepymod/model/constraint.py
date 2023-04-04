"""This module contains concrete implementations of the constraint component."""


import torch
import numpy as np
from .deepmod import Constraint
from typing import List

TensorList = List[torch.Tensor]


class LeastSquares(Constraint):
    def __init__(self) -> None:
        """Least Squares Constraint solved by QR decomposition"""
        super().__init__()

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Calculates the coefficients of the constraint using the QR decomposition for every pair
        of sparse feature matrix and time derivative.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): List of calculated coefficients of size [(n_active_features, 1) x n_outputs].
        """
        coeff_vectors = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            Q, R = torch.qr(theta)  # solution of lst. sq. by QR decomp.
            coeff_vectors.append(torch.inverse(R) @ Q.T @ dt)
        return coeff_vectors


class GradParams(Constraint):
    def __init__(self, n_params: int, n_eqs: int) -> None:
        """Constrains the neural network by optimizing over the coefficients together with the network.
           Coefficient vectors are randomly initialized from a standard Gaussian.

        Args:
            n_params (int): number of features in feature matrix.
            n_eqs (int): number of outputs / equations to be discovered.
        """
        super().__init__()
        self.coeffs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(n_params, 1)) for _ in torch.arange(n_eqs)]
        )

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList):
        """Returns the coefficients of the constraint, since we're optimizing them by
           gradient descent.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): Calculated coefficients of size (n_features, n_outputs).
        """
        return self.coeffs


class Ridge(Constraint):
    """Implements the constraint as a least squares problem solved by QR decomposition."""

    def __init__(self, l=1e-3) -> None:
        super().__init__()
        self.l = torch.tensor(l)

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Calculates the coefficients of the constraint using the QR decomposition for every pair
        of sparse feature matrix and time derivative.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors.
            time_derivs (TensorList): List containing the time derivatives.

        Returns:
            (TensorList): Calculated coefficients.
        """
        coeff_vectors = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            # We use the augmented data method
            norm = torch.norm(theta, dim=0)
            l_normed = torch.diag(
                torch.sqrt(self.l) * norm
            )  # we norm l rather than theta cause we want unnormed coeffs out
            X = torch.cat((theta, l_normed), dim=0)
            y = torch.cat((dt, torch.zeros((theta.shape[1], 1)).to(theta.device)), dim=0)
            # Now solve like normal OLS prob lem
            Q, R = torch.qr(X)  # solution of lst. sq. by QR decomp.
            coeff_vectors.append(torch.inverse(R) @ Q.T @ y)

        return coeff_vectors



class STRidge(Constraint):
    
    def __init__(self, lam = 10**-5, maxit = 100, tol = 0.1, normalize = 2, print_results = False):
        super().__init__()
        self.lam = lam
        self.tol = tol
        self.normalize = normalize
        self.print_results = print_results
        self.maxit = maxit
        
        
    def fit(self, X0: np.ndarray, y: np.ndarray):
        
        """
        STRidge(TrainR,TrainY,lam,STR_iters,tol,normalize = normalize)
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """
        
        n,d = X0.shape
        
        #print("7777777777777777777777777777777777777777")
        #print(n)
        #print(d)
        #print("7777777777777777777777777777777777777777")
        
        
        X = np.zeros((n,d), dtype=np.complex64)
        # First normalize data
        if self.normalize != 0:
            Mreg = np.zeros((d,1))
            for i in range(0,d):
                Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],self.normalize))
                X[:,i] = Mreg[i]*X0[:,i]
        else: X = X0
        
        # Get the standard ridge esitmate
        if self.lam != 0: 
            w = np.linalg.lstsq(X.T.dot(X) + self.lam*np.eye(d),X.T.dot(y),rcond=None)[0]
        else: 
            w = np.linalg.lstsq(X,y,rcond=None)[0]
        num_relevant = d
        biginds = np.where( abs(w) > self.tol)[0]
        
        #print("uuuuuuuuuuuuuuuuuuuuuuuu")
        #print(w)
        #print("uuuuuuuuuuuuuuuuuuuuuuuu")
        
        
        
        # Threshold and continue
        for j in range(self.maxit):

            # Figure out which items to cut out
            smallinds = np.where( abs(w) < self.tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]
                
            # If nothing changes then stop
            if num_relevant == len(new_biginds): break
            else: num_relevant = len(new_biginds)
                
            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0: 
                    #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else: break
            biginds = new_biginds
            
            # Otherwise get a new guess
            w[smallinds] = 0
            
            if self.lam != 0: 
                
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + self.lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
            else: 
                
                w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []: 
            
            w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
        
        if self.normalize != 0: 
            
            return np.multiply(Mreg.reshape(-1,1),w.reshape(-1,1))
        
        else: 
                        
            return [w.reshape(-1,1)]
