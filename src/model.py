"""Access to properties in an SBML model."""

from collections import namedtuple
import pandas as pd  # type: ignore
import tellurium as te  # type: ignore
import libsbml  # type: ignore
from typing import List, Dict


Reaction = namedtuple('Reaction', ['name', 'reactants', 'products', 'kinetic_species'])


class Model(object):
    """CRN Model"""

    def __init__(self, model_str: str):
        """
        Args:
            model_str (str): Antimony model string
        """
        self.roadrunner = te.loadAntimonyModel(model_str)
        sbml_str = self.roadrunner.getSBML()
        reader = libsbml.SBMLReader()
        document = reader.readSBMLFromString(sbml_str)
        self.sbml_model = document.getModel()

    @property
    def species_names(self) -> List[str]:
        """Get list of species names in the model.

        Returns:
            List[str]: Species names
        """
        species_names = [self.sbml_model.getSpecies(i).getId() for i in range(self.sbml_model.getNumSpecies())]
        species_names.sort()
        return species_names
    
    @property
    def reaction_names(self) -> List[str]:
        """Get list of reaction names in the model.

        Returns:
            List[str]: Reaction names
        """
        reaction_names = [self.sbml_model.getReaction(i).getId() for i in range(self.sbml_model.getNumReactions())]
        reaction_names.sort()
        return reaction_names

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