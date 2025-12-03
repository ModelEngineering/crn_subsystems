'''CRN Subsystem'''
from src.model import Model  # type: ignore

import constants as cn  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import sympy as sp  # type: ignore
import tellurium as te  # type: ignore
from typing import List, Optional, Tuple, Dict, cast

DEFAULT_SUBSYSTEM_NAME = "subsystem"


class Subsystem(object):
    '''CRN Subsystem'''

    def __init__(self,
            model_str: str,
            species_names: Optional[List[str]]=None,
            reaction_names: Optional[List[str]]=None,
            subsystem_name: str = DEFAULT_SUBSYSTEM_NAME):
        """
        Args:
            model (str): Antimony model string
            species_names (list-str, optional): species names in subsystem. Default is all species in model
            reaction_names (list-str, optional): reaction names in subsystem. Default is all reactions in model.
            subsystem_name (str, optional): Subsystem name. Defaults to DEFAULT_SUBSYSTEM_NAME.
        """
        self.model = Model(model_str)
        self.species_names = species_names
        self.reaction_names = reaction_names
        self.subsystem_name = subsystem_name

    @property
    def jacobian_smat(self)->sp.Matrix:
        """Get the symboic Jacobian for an LTI system.

        Returns:
            sp.Matrix: Jacobian sympy Matrix
        """
        raise NotImplementedError("jacobian_smat is not implemented yet.")

    def _getImpliedElements(self, species_names: list[str]) -> Tuple[list[str], list[str]]:
        """Get implied species and reactions from selected species and reactions.

        Args:
            species_names (list[str]): Selected species names
            reaction_names (list[str]): Selected reaction names
        Returns:
            tuple[list[str], list[str]]: Implied species and reactions
        """
        reaction_dct = self.model.reaction_dct
        species_set = set(species_names)
        reaction_set = set()
        added = True
        while added:
            added = False
            for reaction_name, reaction in reaction_dct.items():
                if reaction_name in reaction_set:
                    continue
                required_speces = set(reaction.reactants).union(set(reaction.kinetic_species))
                if required_speces.issubset(species_set):
                    reaction_set.add(reaction_name)
                    species_set.update(reaction.kinetic_species)
                    added = True
        return list(species_set), list(reaction_set)
    
    def calculateStepResponse(self, initial_condition_dct: Optional[Dict[str, float]]=None) -> pd.DataFrame:
        """Compute the step response of the subsystem.

        Args:
            initial_condition_dct (Optional[Dict[str, float]], optional): Initial conditions for species. Defaults to None.

        Returns:
            pd.DataFrame: Step response data
                rows: input species
                columns: output species
        """
        if initial_condition_dct is None:
            initial_condition_dct = {}
        # Do initialzations
        self.model.roadrunner.resetToOrigin()
        self.model.roadrunner.reset()
        for species, value in initial_condition_dct.items():
            self.model.roadrunner[species] = value
        # Calculate the step response
        steady_state_arr = self.model.roadrunner.getSteadyStateValues()
        df = pd.DataFrame(steady_state_arr, columns=self.model.roadrunner.getFloatingSpeciesIds())
        df = df[cast(List[str], self.species_names)]
        return df
    
    def calculateEigenvalues(self) -> np.ndarray:
        """Compute the eigenvalues of the subsystem's Jacobian at steady state.

        Returns:
            np.ndarray: Eigenvalues of the Jacobian
        """
        J = self.model.roadrunner.getFullJacobian()
        eigvals = np.linalg.eigvals(J)
        return eigvals  