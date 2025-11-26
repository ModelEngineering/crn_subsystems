'''Creates a symbolic Jacobian matrix from an Antimony model.'''

from collections import namedtuple
import libsbml # type: ignore
import sympy as sp  # type: ignore
import tellurium as te  # type: ignore
from collections import OrderedDict
from typing import Tuple, Dict, List, Any


ReactionSymbolicJacobian = namedtuple("ReactionSymbolicJacobian", ["A_smat", "b_smat",
        "reactant_name", "product_names", "kinetic_constant_name"])
ReactionDescription = namedtuple("ReactionDescription",
        ["reaction_name", "reactant_name", "kinetic_constant_name", "product_stoichiometry_dct"]) 
MakeSymbolicJacobianResult = namedtuple("MakeSymbolicJacobianResult",
        ["jacobian_smat", "kinetic_constant_dct", "species_names"])
#  jacobian_mat : sympy.Matrix
#      Symbolic Jacobian matrix where J[i,j] = d(rate_i)/d(spec
#  jacobian_smat : sympy.Matrix
#      Sparse symbolic Jacobian matrix
#  kinetic_constant_dict : dict
#      Dictionary mapping parameter names to their values

class SymbolicJacobianMaker(object):

    def __init__(self, antimony_str: str):
        """
        Convert an Antimony model to a symbolic Jacobian matrix.
        Reactions must either have 0 or 1 reactant with unit stoichiometry.
        Kinetics are assumed to be mass action (k*S) if 1 reactant
        
        Parameters:
        -----------
        antimony_str: String for an Antimony model
            Path to Antimony file
        """
        self.antimony_str = antimony_str
        # See build() for other attributes

    def initialize(self)->None:
        # Initializes the object
        self.model, self.roadrunner = self._makeModel()
        self.jacobian_mat = self.roadrunner.getFullJacobian()
        self.kinetic_constant_dct: dict = self._makeKineticConstantDct()
        self.species_names: list = self._makeSpeciesNames()
        self.num_species: int = len(self.species_names)
        self.jacobian_smat, self.b_smat = self._makeSymbolicJacobian()

    def _getSpeciesIndex(self, species_name: str)->int:
        # Get the index of a species in the species_names list
        try:
            idx = self.species_names.index(species_name)
        except ValueError:
            raise ValueError(f"Species {species_name} not found in species names")
        return idx

    def _makeModel(self)->Tuple[libsbml.Model, Any]:
        # Load with tellurium and convert to SBML
        rr = te.loada(self.antimony_str)
        sbml_str = rr.getSBML()
        # Get the SBML model
        reader = libsbml.SBMLReader()
        document = reader.readSBMLFromString(sbml_str)
        if document.getNumErrors() > 0:
            print("Errors in SBML conversion:")
            document.printErrors()
            raise ValueError("Failed to convert Antimony to SBML")
        model = document.getModel()
        return model, rr
    
    def _makeSpeciesNames(self)->List[str]:
        # Include all species, including boundary/constant species
        species_names = []
        # Get species and create symbols
        for i in range(self.model.getNumSpecies()):
            species = self.model.getSpecies(i)
            species_name = species.getId()
            species_names.append(species_name)
        species_names.sort()
        return species_names
    
    def _makeKineticConstantDct(self)->Dict[str, float]:
        kinetic_constant_dct = {}
        # Get parameters and create symbols and dictionary
        species_names = self._makeSpeciesNames()
        for i in range(self.model.getNumParameters()):
            param = self.model.getParameter(i)
            param_id = param.getId()
            if param_id in species_names:
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
        for i in range(self.model.getNumReactions()):
            reaction = self.model.getReaction(i)
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
            species_names = findStrs(kinetic_law_strs, self.species_names)
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
        reactions = [self.model.getReaction(i) for i in range(self.model.getNumReactions())]
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
        The B matrix is n X 1, and contains the rate at which the i-th species changes

        Args:
            reaction (libsbml.Reaction): _description_

        Returns:
            SymbolicJacobian
        """
        # Extract the rate constant and species for each reaction
        ##
        def findStrs(candidates: List[str], targets: List[str]) -> List[str]:
            # Find the candidates that are equal to the targets
            results = set(candidates).intersection(targets)
            return list(results)
        ##
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
        kinetic_law_strs = [s.strip() for s in kinetic_law_str.formula.split(" ")]
        kinetic_constants = findStrs(kinetic_law_strs, list(self.kinetic_constant_dct.keys()))
        if len(kinetic_constants) != 1:
            raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
        kinetic_constant_name = kinetic_constants[0]
        species_names = findStrs(kinetic_law_strs, self.species_names)
        if len(species_names) > 1:
            raise ValueError(f"Reaction {reaction.getId()} has unexpected kinetic law: {kinetic_law_str}")
        kinetic_species_name = species_names[0] if len(species_names) == 1 else None
        # Get products and their stoichiometries
        product_stoichiometry_dct: dict = {s: 0 for s in self.species_names}  # key: product_id, value: stoichiometry
        for j in range(reaction.getNumProducts()):
            product = reaction.getProduct(j)
            product_id = product.getSpecies()
            product_stoichiometry_dct[product_id] += product.getStoichiometry()
        # Construct the symbolic Jacobian for this reaction
        A_smat = sp.Matrix.zeros(len(self.species_names), len(self.species_names))
        b_smat = sp.Matrix.zeros(len(self.species_names), 1)
        kinetic_constant_symbol = sp.Symbol(kinetic_constant_name)
        #   Handle the reactant
        if reactant_name is not None:
            reactant_idx = self._getSpeciesIndex(reactant_name)
            if kinetic_species_name is not None:
                kinetic_species_idx = self._getSpeciesIndex(kinetic_species_name)
                A_smat[reactant_idx, kinetic_species_idx] = -kinetic_constant_symbol
            else:
                b_smat[reactant_idx] = -kinetic_constant_symbol
        #   Handle the products
        if kinetic_species_name is not None:
            kinetic_species_idx = self._getSpeciesIndex(kinetic_species_name)
        else:
            kinetic_species_idx = -1
        for product_id, stoich in product_stoichiometry_dct.items():
            if stoich == 0:
                continue
            product_idx = self._getSpeciesIndex(product_id)
            if kinetic_species_name is not None:
                A_smat[product_idx, kinetic_species_idx] += stoich * kinetic_constant_symbol
            else:
                b_smat[product_idx] += stoich * kinetic_constant_symbol
        #
        product_names = [p for p, s in product_stoichiometry_dct.items() if s > 0]
        return ReactionSymbolicJacobian(A_smat=A_smat, b_smat=b_smat, reactant_name=reactant_name,
                product_names=product_names, kinetic_constant_name=kinetic_constant_name)