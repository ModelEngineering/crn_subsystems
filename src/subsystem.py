'''CRN Subsystem in terms of species and reactions. Calculate properties. Do operations.'''

import src.constants as cn  # type: ignore
from src.jacobian import Jacobian  # type: ignore
from src.model import Model  # type: ignore
from src import util  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
import sympy as sp  # type: ignore
import tellurium as te  # type: ignore
from typing import List, Optional, Tuple, Dict, cast

DEFAULT_SUBSYSTEM_NAME = "subsystem"

# TODO: tests


class Subsystem(object):
    '''CRN Subsystem'''

    def __init__(self,
            model_str: str,
            species_names: Optional[List[str]]=None,
            reaction_names: Optional[List[str]]=None,
            subsystem_name: str = DEFAULT_SUBSYSTEM_NAME,
            time: float = 0):
        """
        Args:
            model (str): Antimony model string
            species_names (list-str, optional): species names in subsystem. Default is all species in model
            reaction_names (list-str, optional): reaction names in subsystem. Default is all reactions in model.
            subsystem_name (str, optional): Subsystem name. Defaults to DEFAULT_SUBSYSTEM_NAME.
            time (float, optional): Time at which to evaluate the Jacobian. Defaults to 0. infty is steady steate
        """
        self.subsystem_name = subsystem_name
        self.time = time
        #
        self.model = Model(model_str, species_names=species_names,
                reaction_names=reaction_names)
        self.jacobian = Jacobian(self.model, time=time)

    def calculateEigenvalues(self) -> np.ndarray:
        """Calculate the eigenvalues of the Jacobian.

        Args:
            time (float, optional): Time at which to evaluate the Jacobian. Defaults to

        Returns:
            np.ndarray: Eigenvalues
        """
        jacobian_mat = self.jacobian.jacobian_df.to_numpy()
        eigenvalues = np.linalg.eigvals(jacobian_mat)
        return eigenvalues
    
    def calculateNumericDiscs(self) -> np.ndarray:
        """
        Calculates Gershgorinn discs for the continuous-time eigenvalues.

        Returns:
            np.ndarray: A matrix containing the lower and upper bounds of
                continuous-time Girgorian discs
        """
        matrix = self.jacobian.jacobian_df.to_numpy()
        # Extract diagonal elements
        diagonal = np.diag(matrix)
        # Compute sum of absolute values of off-diagonal elements for each row
        off_diagonal_sums = np.zeros(self.model.num_species)
        for i in range(self.model.num_species):
            row_abs = np.abs(matrix[i, :])
            off_diagonal_sums[i] = np.sum(row_abs) - np.abs(diagonal[i])
        # Calculate the bounds
        lower_bounds = diagonal - off_diagonal_sums
        upper_bounds = diagonal + off_diagonal_sums
        #
        return np.hstack([lower_bounds, upper_bounds]).reshape(2, self.model.num_species).T
    
    def isStable(self) -> bool:
        """
        Determines if the subsystem is stable (all eigenvalues have negative real part).

        Returns:
            bool: True if stable, False otherwise
        """
        eigenvalues = self.calculateEigenvalues()
        return all(np.real(eig) < 0 for eig in eigenvalues)

    def calculateSymbolicDiscs(self) -> sp.Matrix:
        """
        Calculates symbolic Gershgorinn discs for the continuous-time eigenvalues.

        Returns:
            sp.Matrix: A matrix containing the lower and upper bounds of
                continuous-time Girgorian discs
        """
        matrix = self.jacobian.jacobian_smat
        # Extract diagonal elements
        diagonal = matrix.diagonal()
        # Compute sum of absolute values of off-diagonal elements for each row
        off_diagonal_sums = sp.Matrix.zeros(self.model.num_species, 1)
        for i in range(self.model.num_species):
            row_0_L1 = sp.Add(*[sp.Abs(x) for x in matrix.row(i)])
            off_diagonal_sums[i] = row_0_L1 - sp.Abs(diagonal[i])  # type: ignore
        # Calculate the bounds
        off_diagonal_sums = off_diagonal_sums.applyfunc(sp.simplify)
        off_diagonal_sums = off_diagonal_sums.reshape(1, self.model.num_species)
        lower_bounds = diagonal - off_diagonal_sums
        upper_bounds = diagonal + off_diagonal_sums
        #
        # FIXME: wrong shape
        smat = sp.Matrix.hstack(lower_bounds, upper_bounds).reshape(2, self.model.num_species).T
        return smat

    def calculateNumericStepResponse(self, input_species_name: str,
            step_size: float = 1.0) -> pd.Series:
        """
        Calculates the step response using steady state analysis by setting all derivatives to zero.
        to zero and requiring that the input species increases by step_size.

        Args:
            time_vec (np.ndarray): Time vector for the step response.
            step_size (float): The magnitude of the step input.

        Returns:
            step response (float): ratio of output to input
        """
        b_mat = np.zeros((self.model.num_species))
        b_mat[self.model.getSpeciesIndex(input_species_name)] = step_size
        A_mat = np.array(self.jacobian.jacobian_df.values, dtype=float)
        species_idx = self.model.getSpeciesIndex(input_species_name)
        A_mat[species_idx, :] = np.zeros((1, self.model.num_species))
        # Solve Ax = -b for steady state
        xss_arr, residual, _ = util.solveLinearSystem(A_mat, b_mat, 
                fixed={self.model.getSpeciesIndex(input_species_name): step_size})
        if residual > 1e-6:
            #raise ValueError(f"Could not solve for steady state: residual={residual}")
            pass
        xss_ser = pd.Series(xss_arr, index=self.model.species_names)
        step_response_ser = xss_ser[self.output_species_name] / step_size  # type: ignore
        return step_response_ser
    
    def calculateSymbolicStepResponse(self, species_name: str) -> Dict[str, sp.Expr]:
        """
        Calculates the symbolic step response for the specified input species.
        At steady state, all derivatives are zero and the input_species is stepped by 1 unit.
        0 = A * x_ss + b, where x_ss is the steady-state concentrations after the step.
        Or, - A[:, 1] - b = A[:, :1] * x_ss[1:]


        Args:
            species_name (str): Input species name

        Returns:
            pd.DataFrame: Step response data
                rows: input species
                columns: output species
        """
        species_idx = self.model.getSpeciesIndex(species_name)
        b_smat = self.jacobian.b_smat
        # Construct the A' matrix and c vector
        Ap_smat = self.jacobian.jacobian_smat.copy()
        Ap_smat[species_idx, :] = sp.Matrix.zeros(1, self.model.num_species)
        Ap_smat = Ap_smat[:, [i for i in range(self.model.num_species) if i != species_idx]]
        b_smat[species_idx] = 0
        c_smat = self.jacobian.jacobian_smat[:, species_idx]
        c_smat = sp.Matrix.zeros(self.model.num_species, 1) - c_smat
        # Solve for steady-state
        x_solns = [sp.Symbol(n) for n in self.model.species_names if n != species_name]
        #x2_ss,x3_ss, x4_ss = sp.symbols("x2_ss,x3_ss, x4_ss")
        #x_ss = sp.Matrix([x2_ss, x3_ss, x4_ss])
        x_ss = sp.Matrix(x_solns)
        symbolic_solution_dct = sp.solve(Ap_smat * x_ss + b_smat - c_smat, x_solns)
        solution_dct = {k.name: v for k, v in symbolic_solution_dct.items()}
        solution_dct.update({species_name: 1})
        return solution_dct
    
    def union(self, other: 'Subsystem') -> 'Subsystem':
        """Create a new Subsystem that is the union of this and another Subsystem.

        Args:
            other (Subsystem): Another Subsystem
        Returns:
            Subsystem: New Subsystem representing the union
        """
        raise NotImplementedError("union is not implemented yet.")
    
    def difference(self, other: 'Subsystem') -> 'Subsystem':
        """Create a new Subsystem that is the difference of this and another Subsystem.

        Args:
            other (Subsystem): Another Subsystem
        Returns:
            Subsystem: New Subsystem representing the difference    
        """
        raise NotImplementedError("difference is not implemented yet.")