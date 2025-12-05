"""Access to properties in an SBML model."""

from collections import namedtuple
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore
import libsbml  # type: ignore
from typing import List, Dict, Optional

DUMMY_REACTANT = "DUMMY_REACTANT"
DUMMY_PRODUCT = "DUMMY_PRODUCT"
DUMMY_KINETIC_SPECIES = "DUMMY_KINETIC_SPECIES"


Reaction = namedtuple('Reaction', ['name', 'reactants', 'products', 'kinetic_species'])


class Model(object):
    """CRN Model"""

    def __init__(self, model_str: str,
            species_names: Optional[List[str]]=None,
            reaction_names: Optional[List[str]]=None):
        """
        Args:
            model_str (str): Antimony model string
            species_names (Optional[List[str]]): List of species names
            reaction_names (Optional[List[str]]): List of reaction names
        """
        roadrunner = te.loadAntimonyModel(model_str)
        sbml_str = roadrunner.getSBML()
        reader = libsbml.SBMLReader()
        document = reader.readSBMLFromString(sbml_str)
        sbml_model = document.getModel()
        #
        if species_names is None:
            species_names = [sbml_model.getSpecies(i).getId()
                    for i in range(sbml_model.getNumSpecies())]
        self.species_names = species_names
        if reaction_names is None:
            reaction_names = [sbml_model.getReaction(i).getId()
                    for i in range(sbml_model.getNumReactions())]
        self.reaction_names = reaction_names 
        # Update the SBML model
        self.sbml_model = self._makeConstrainedSBMLModel(sbml_model)
        self.antimony_str = self.makeAntimony()
        self.roadrunner = te.loada(self.antimony_str)

    @staticmethod
    def _makeSpecies(sbml_model: libsbml.Model, species_name: str,
            is_constant: bool=True, initial_concentration: float=0.0    ) -> libsbml.Species:
        """Creates a new species in the SBML model.

        Args:
            species_name (str): Species name
        """
        new_species = sbml_model.createSpecies()
        new_species.setId(species_name)
        new_species.setCompartment("default")
        new_species.setBoundaryCondition(False)
        new_species.setConstant(is_constant)
        new_species.setHasOnlySubstanceUnits(False)
        new_species.setInitialConcentration(initial_concentration)
        return new_species
    
    def _updateSpeciesInKineticLaw(self, sbml_model: libsbml.Model, reaction: libsbml.Reaction):
        """Make all occurrences of excluded species in a reaction's kinetic law as constants"""
        kinetic_law = reaction.getKineticLaw()
        math_ast = kinetic_law.getMath()
        ##
        def traverse_and_replace(node):
            # Recursive function to traverse and replace species
            if node.isName():
                node_name = node.getName()
                if not node_name in self.species_names:
                    species = sbml_model.getSpecies(node_name)
                    if species is None:
                        return
                    if not species.getConstant():
                        # Replace with dummy kinetic species
                        self._setConstantSpecies(sbml_model, node_name)
                    return
            else:
                for i in range(node.getNumChildren()):
                    traverse_and_replace(node.getChild(i))
            return
        ##
        traverse_and_replace(math_ast)

    def _makeConstrainedSBMLModel(self, sbml_model: libsbml.Model) -> libsbml.Model:
        """Updates the SBML model to create constant, boundary species for excluded species.
        Removed species are set to a constant 0.

        Args:
            sbml_model (libsbml.Model): Original SBML model

        Returns:
            libsbml.Model: Subset SBML model
        """
        new_sbml_model = sbml_model.clone()
        # Add species
        for i in range(sbml_model.getNumSpecies()):
            species = sbml_model.getSpecies(i)
            if not species.getId() in self.species_names:
                self._setConstantSpecies(new_sbml_model, species.getId())
        # Add reactions
        for i in range(sbml_model.getNumReactions()):
            reaction = sbml_model.getReaction(i)
            if not reaction.getId() in self.reaction_names:
                new_sbml_model.removeReaction(reaction.getId())
            else:
                # Edit reactants
                for j in range(reaction.getNumReactants()):
                    reactant = reaction.getReactant(j)
                    if reactant.getSpecies() not in self.species_names:
                        species_name = reactant.getSpecies()
                        self._setConstantSpecies(new_sbml_model, species_name)
                # Edit products
                for j in range(reaction.getNumProducts()):
                    product = reaction.getProduct(j)
                    if product.getSpecies() not in self.species_names:
                        species_name = product.getSpecies()
                        self._setConstantSpecies(new_sbml_model, species_name)
                # Update species in kinetic law
                self._updateSpeciesInKineticLaw(new_sbml_model, reaction)
        #
        return new_sbml_model
    
    def _setConstantSpecies(self, sbml_model: libsbml.Model, species_name: List[str]):
        """Set specified species as constant in the SBML model.

        Args:
            sbml_model (libsbml.Model): SBML model
            species_name str: List of species names to set as constant
        """
        species = sbml_model.getSpecies(species_name)
        species.setConstant(True)
        species.setBoundaryCondition(True)
        species.setInitialConcentration(0.0)

    @staticmethod 
    def _getSpeciesFromKineticLaw(reaction: libsbml.Reaction, model: libsbml.Model) -> list[str]:
        kl = reaction.getKineticLaw()
        if kl is None:
            return []

        formula = libsbml.formulaToString(kl.getMath())
        species_ids = {model.getSpecies(i).getId() 
                    for i in range(model.getNumSpecies())}
        # Check which species appear in the formula string
        return [sp for sp in species_ids if sp in formula]

    @property 
    def reaction_dct(self) -> Dict[str, Reaction]:
        """
        Create a dictionary representation of reactions in the model.

        Parameters
        ----------
        model: str
            Antimony model string

        Returns
        -------
        Dict[str, Reaction]
            Dictionary mapping reaction names to Reaction namedtuples
        """

        reaction_dct: Dict[str, Reaction] = {}

        for i in range(self.sbml_model.getNumReactions()):
            sbml_reaction = self.sbml_model.getReaction(i)
            reaction_name = sbml_reaction.getId()

            reactants = []
            for j in range(sbml_reaction.getNumReactants()):
                species_ref = sbml_reaction.getReactant(j)
                species_id = species_ref.getSpecies()
                reactants.append(species_id)

            products = []
            for j in range(sbml_reaction.getNumProducts()):
                species_ref = sbml_reaction.getProduct(j)
                species_id = species_ref.getSpecies()
                products.append(species_id)

            kinetic_species = self._getSpeciesFromKineticLaw(sbml_reaction, self.sbml_model)

            reaction = Reaction(
                name=reaction_name,
                reactants=reactants,
                products=products,
                kinetic_species=kinetic_species
            )
            reaction_dct[reaction_name] = reaction

        return reaction_dct
    
    def makeSBML(self) -> str:
        """Returns the SBML string of the model.

        Returns:
            str: SBML string
        """
        document = libsbml.SBMLDocument(2, 1)
        # You can also check the model's level/version
        if self.sbml_model.getLevel() and self.sbml_model.getVersion():
            document = libsbml.SBMLDocument(self.sbml_model.getLevel(), self.sbml_model.getVersion())
        else:
            document = libsbml.SBMLDocument(3, 1)  # Default to Level 3, Version 1
        document.setModel(self.sbml_model)
        writer = libsbml.SBMLWriter()
        sbml_str = writer.writeSBMLToString(document)
        return sbml_str
    
    def makeAntimony(self) -> str:
        """Returns the Antimony string of the model.

        Returns:
            str: Antimony string
        """
        sbml_str = self.makeSBML()
        roadrunner = te.loadSBMLModel(sbml_str)
        antimony_str = roadrunner.getAntimony()
        return antimony_str