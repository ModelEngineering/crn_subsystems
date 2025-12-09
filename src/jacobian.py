"""Constructs and manipulates Jacobians"""

from src.model import Model

from collections import namedtuple
import libsbml # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import sympy as sp  # type: ignore
from typing import Dict, Tuple

# TODO: Tests for different time options

_ReactionSymbolicJacobian = namedtuple("_ReactionSymbolicJacobian", ["A_smat", "b_smat"])

class Jacobian(object):

    def __init__(self, model: Model, time: float = 0):
        """Initializes Jacobian object.

        Args:
            model (Model): CRN Model. Assumes that input species are not fixed, boundaries.
            time (float, optional): Time at which to evaluate the Jacobian. Defaults to 0.
        """
        self.model = model
        self.model.roadrunner.reset()
        if time == 0:
            pass
        elif time == np.inf:
            self.model.roadrunner.steadyState()
        else:
            self.model.roadrunner.simulate(0, time, 2)
        #
        jacobian_mat = self.model.roadrunner.getFullJacobian()
        self.jacobian_df = pd.DataFrame(
            jacobian_mat,
            index=jacobian_mat.rownames,
            columns=jacobian_mat.colnames
        )
        self.jacobian_df = self.jacobian_df[self.model.species_names]
        self.jacobian_smat, self.b_smat = self._makeSymbolicJacobian()

    def copy(self) -> 'Jacobian':
        """Creates a copy of the Jacobian object.

        Returns:
            Jacobian: Copy of the Jacobian object
        """
        return Jacobian(self.model)
    
    def _makeSymbolicJacobian(self)->Tuple[sp.Matrix, sp.Matrix]:
        # Create the symbolic Jacobian matrix and the b matrix
        # Returns: (jacobian_smat, b_smat)
        reactions = [self.model.libsbml_model.getReaction(i) for i in range(self.model.libsbml_model.getNumReactions())]
        lti_results = [self._makeReactionLti(r) for r in reactions]
        symbolic_jacobians = [r.A_smat for r in lti_results]
        symbolic_bs = [r.b_smat for r in lti_results]
        jacobian_smat = symbolic_jacobians[0]
        for smat in symbolic_jacobians[1:]:
            jacobian_smat += smat
        b_smat = symbolic_bs[0]
        for smat in symbolic_bs[1:]:
            b_smat += smat
        return jacobian_smat, b_smat

    def _makeReactionLti(self, reaction: libsbml.Reaction)->_ReactionSymbolicJacobian:
        """ Creates the symbolic A and B LTI matrices for a given reaction.
        This is done in general by differentiaing the rate law with respect to each species.
        The B matrix is n X 1, and contains the rate at which the i-th species changes

        Args:
            reaction (libsbml.Reaction)

        Returns:
            SymbolicJacobian
            Symbolic b matrix
        """
        # Extract the rate constant and species for each reaction
        A_smat = sp.Matrix.zeros(len(self.model.species_names), len(self.model.species_names))
        b_smat = sp.Matrix.zeros(len(self.model.species_names), 1)
        # Create symbols for all species and parameters
        species_sp_dct = {s: sp.Symbol(s) for s in self.model.species_names}
        parameter_sp_dct = {p: sp.Symbol(p) for p in self.model.parameter_dct.keys()}
        # Change the kinetic law to a sympy expression and take derivatives
        kinetic_law_str = reaction.getKineticLaw()
        kinetic_law_expr = sp.parse_expr(kinetic_law_str.formula, local_dict={**species_sp_dct, **parameter_sp_dct})
        derivative_dct: Dict[str, sp.Expr] = {}
        for species_name in self.model.species_names:
            derivative_dct[species_name] = sp.diff(kinetic_law_expr, species_sp_dct[species_name])
        is_constant = all([derivative_dct[s] == 0 for s in self.model.species_names])
        # Process the reactants
        num_reactants = reaction.getNumReactants()
        for icol in range(self.model.num_species):  # Possible kinetic species
            species_name = self.model.species_names[icol]
            derivative = derivative_dct[species_name]
        # Process reactants
        for ireactant in range(num_reactants):
            reactant = reaction.getReactant(ireactant)
            reactant_name = reactant.getSpecies()
            reactant_stoich = reactant.getStoichiometry()
            reactant_idx = self.model.getSpeciesIndex(reactant_name)
            for icol in range(self.model.num_species):  # Possible kinetic species
                species_name = self.model.species_names[icol]
                A_smat[reactant_idx, icol] -= reactant_stoich * derivative_dct[species_name]
            if is_constant:
                b_smat[reactant_idx] -= reactant_stoich * kinetic_law_expr
        # Process products
        for iproduct in range(reaction.getNumProducts()):
            product = reaction.getProduct(iproduct)
            product_name = product.getSpecies()
            product_stoich = product.getStoichiometry()
            product_idx = self.model.getSpeciesIndex(product_name)
            for icol in range(self.model.num_species):  # Possible kinetic species
                species_name = self.model.species_names[icol]
                A_smat[product_idx, icol] += product_stoich * derivative_dct[species_name]
            if is_constant:
                b_smat[product_idx] += product_stoich * kinetic_law_expr
        #
        return _ReactionSymbolicJacobian(A_smat=A_smat, b_smat=b_smat)