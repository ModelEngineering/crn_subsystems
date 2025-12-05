"""Constructs and manipulates Jacobians"""

from src.model import Model
from src.symbolic_jacobian_maker import SymbolicJacobianMaker  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
from typing import Dict


class Jacobian(object):

    def __init__(self, model: Model):
        """Initializes Jacobian object.

        Args:
            model (Model): CRN Model. Assumes that input species are not fixed, boundaries.
        """
        self.model = model

    def copy(self) -> 'Jacobian':
        """Creates a copy of the Jacobian object.

        Returns:
            Jacobian: Copy of the Jacobian object
        """
        return Jacobian(self.model)

    @property
    def jacobian_df(self) -> pd.DataFrame:
        jacobian_arr = self.model.roadrunner.getFullJacobian()
        species_names = jacobian_arr.rownames
        df = pd.DataFrame(jacobian_arr, columns=species_names, index=species_names)
        sorted_df = df.loc[self.model.species_names, self.model.species_names]
        return sorted_df
    
    def _makeSymbolicJacobian(self) -> pd.DataFrame:
        """Get the symbolic Jacobian DataFrame of the subsystem.

        Returns:
            pd.DataFrame: Symbolic Jacobian DataFrame
                rows: species names
                columns: species names
        """
        raise NotImplementedError("_makeSymbolicJacobian is not implemented yet." )
    
    def calculateEigenvalues(self) -> np.ndarray:
        """Calculate the eigenvalues of the Jacobian.

        Returns:
            np.ndarray: Eigenvalues
        """
        jacobian_mat = self.jacobian_df.to_numpy()
        eigenvalues = np.linalg.eigvals(jacobian_mat)
        return eigenvalues
    
    def calculateStepResponse(self, input_dct: Dict[str, float]) -> pd.DataFrame:
        """Compute the step response of the subsystem.

        Args:
            input_dct (Dict[str, float]): Input species and their step values

        Returns:
            pd.DataFrame: Step response data
                rows: input species
                columns: output species
        """
        raise NotImplementedError("calculateStepResponse is not implemented yet.")
