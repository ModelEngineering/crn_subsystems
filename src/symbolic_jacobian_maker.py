'''Creates a symbolic Jacobian matrix from an Antimony model.'''
from src.model import Model  # type: ignore

from collections import namedtuple
import libsbml # type: ignore
import sympy as sp  # type: ignore
import tellurium as te  # type: ignore
from collections import OrderedDict
from typing import Tuple, Dict, List, Any


ReactionSymbolicJacobian = namedtuple("ReactionSymbolicJacobian", ["A_smat", "b_smat"])
ReactionDescription = namedtuple("ReactionDescription",
        ["reaction_name", "reactant_name", "kinetic_constant_name", "product_stoichiometry_dct"]) 


class SymbolicJacobianMaker(object):

    def __init__(self, model: Model):
        """
        Convert an Antimony model to a symbolic Jacobian matrix.
        Reactions must either have 0 or 1 reactant with unit stoichiometry.
        Kinetics are assumed to be mass action (k*S) if 1 reactant
        
        Parameters:
        -----------
            model (Model): CRN Model
        """
        self.model = model

    def initialize(self)->None:
        # Initializes the object
        self.jacobian_mat = self.model.roadrunner.getFullJacobian()
        self.kinetic_constant_dct: dict = self._makeKineticConstantDct()
        self.num_species: int = len(self.model.species_names)
        self.jacobian_smat, self.b_smat = self._makeSymbolicJacobian()

    def _getSpeciesIndex(self, species_name: str)->int:
        # Get the index of a species in the species_names list
        try:
            idx = self.model.species_names.index(species_name)
        except ValueError:
            raise ValueError(f"Species {species_name} not found in species names")
        return idx
    
    def _makeKineticConstantDct(self)->Dict[str, float]:
        kinetic_constant_dct = {}
        # Get parameters and create symbols and dictionary
        for i in range(self.model.libsbml_model.getNumParameters()):
            param = self.model.libsbml_model.getParameter(i)
            param_id = param.getId()
            if param_id in self.model.species_names:
                continue
            param_value = param.getValue()
            kinetic_constant_dct[param_id] = param_value
        return kinetic_constant_dct

    def _makeReactionDescription(self)->Dict[str, ReactionDescription]:
        # Extract the rate constant and species for each reaction
        ##
        def findStrs(candidates: List[str], targets: List[str]) -> List[str]:
            # Find the candidates that are equal to the targets
            results = set(candidates).intersection(targets)
            return list(results)
        ##
        reaction_description_dct: Dict[str, ReactionDescription] = {}
        for i in range(self.model.libsbml_model.getNumReactions()):
            reaction = self.model.libsbml_model.getReaction(i)
            reaction_name = reaction.getId()
            # Get reactants (should be 0 or 1)
            num_reactants = reaction.getNumReactants()
            if num_reactants > 1:
                raise ValueError(f"Reaction {reaction_name} has {num_reactants} reactants. Only 0 or 1 allowed.")
            reactant_name = None
            if num_reactants == 1:
                reactant = reaction.getReactant(0)
                if reactant.getStoichiometry() != 1:
                    raise ValueError(f"Reaction {reaction_name} has non-unit stoichiometry")
                reactant_name = reactant.getSpecies()
            # Parse the kinetic law to extract the rate constant, products, and their stoichiometries
            kinetic_law_str = reaction.getKineticLaw()
            kinetic_law_strs = kinetic_law_str.formula.split(" ").strip()
            kinetic_constants = findStrs(kinetic_law_strs, list(self.kinetic_constant_dct.keys()))
            if len(kinetic_constants) != 1:
                raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
            kinetic_constant_name = kinetic_constants[0]
            species_names = findStrs(kinetic_law_strs, self.model.species_names)
            if len(species_names) > 1:
                raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
            # Get products and their stoichiometries
            product_stoichiometry_dct: dict = {}  # key: product_id, value: stoichiometry
            for j in range(reaction.getNumProducts()):
                product = reaction.getProduct(j)
                product_id = product.getSpecies()
                product_stoichiometry_dct[product_id] = product.getStoichiometry()
            # Create ReactionDescription
            reaction_description_dct[reaction_name] = ReactionDescription(
                reaction_name=reaction_name,
                reactant_name=reactant_name,
                kinetic_constant_name=kinetic_constant_name,
                product_stoichiometry_dct=product_stoichiometry_dct)
        #
        return reaction_description_dct
    
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

    def _makeReactionLti(self, reaction: libsbml.Reaction)->ReactionSymbolicJacobian:
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
            reactant_idx = self._getSpeciesIndex(reactant_name)
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
            product_idx = self._getSpeciesIndex(product_name)
            for icol in range(self.model.num_species):  # Possible kinetic species
                species_name = self.model.species_names[icol]
                A_smat[product_idx, icol] += product_stoich * derivative_dct[species_name]
            if is_constant:
                b_smat[product_idx] += product_stoich * kinetic_law_expr
        #
        return ReactionSymbolicJacobian(A_smat=A_smat, b_smat=b_smat)