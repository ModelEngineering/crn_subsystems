"""Merge two libsbml documents into a single model."""

import libsbml  # type: ignore
import tellurium as te  # type: ignore
import numpy as np
from typing import Dict, Set, Union



def mergeModels(model1_str: str, model2_str: str) -> str:
    """
    Merge two libsbml documents into a single document containing all species and reactions.

    Args:
        model1_str: First model
        model2_str: Second model

    Returns:
        libsbml.SBMLDocument: Merged document containing all species and reactions from both models

    Notes:
        - If species/reactions/parameters have the same ID in both models, values from doc1 take precedence
        - Compartments are merged, with doc1 values taking precedence for duplicates
        - Both models should have compatible SBML levels/versions
    """
    rr = te.loadAntimonyModel(model1_str)
    reader = libsbml.SBMLReader()
    doc1 = reader.readSBMLFromString(rr.getSBML())
    model1 = doc1.getModel()
    rr = te.loadAntimonyModel(model2_str)
    reader = libsbml.SBMLReader()
    doc2 = reader.readSBMLFromString(rr.getSBML())
    model2 = doc2.getModel()

    # Create new document with same level/version as doc1
    level = doc1.getLevel() if doc1.getLevel() else 3
    version = doc1.getVersion() if doc1.getVersion() else 1
    merged_doc = libsbml.SBMLDocument(level, version)
    merged_model = merged_doc.createModel()

    # Set model ID (use doc1's ID or create a default)
    model_id = doc1.getId() if doc1.getId() else "merged_model"
    merged_model.setId(model_id)

    # Track what we've added to avoid duplicates
    added_compartments: Set[str] = set()
    added_species: Set[str] = set()
    added_parameters: Set[str] = set()
    added_reactions: Set[str] = set()

    # Merge compartments from both models
    for model in [model1, model2]:
        for i in range(model.getNumCompartments()):
            comp = model.getCompartment(i)
            comp_id = comp.getId()
            if comp_id not in added_compartments:
                new_comp = merged_model.createCompartment()
                new_comp.setId(comp_id)
                new_comp.setConstant(comp.getConstant())
                new_comp.setSpatialDimensions(comp.getSpatialDimensions())
                if comp.isSetSize():
                    new_comp.setSize(comp.getSize())
                added_compartments.add(comp_id)

    # If no compartments were added, create a default one
    if len(added_compartments) == 0:
        default_comp = merged_model.createCompartment()
        default_comp.setId("default")
        default_comp.setConstant(True)
        default_comp.setSpatialDimensions(3)
        default_comp.setSize(1.0)
        added_compartments.add("default")

    # Merge species from both models
    for model in [model1, model2]:
        for i in range(model.getNumSpecies()):
            species = model.getSpecies(i)
            species_id = species.getId()
            if species_id not in added_species:
                new_species = merged_model.createSpecies()
                new_species.setId(species_id)

                # Set compartment (use existing or default to first available)
                comp_id = species.getCompartment()
                if comp_id in added_compartments:
                    new_species.setCompartment(comp_id)
                else:
                    new_species.setCompartment(list(added_compartments)[0])

                new_species.setConstant(species.getConstant())
                new_species.setBoundaryCondition(species.getBoundaryCondition())
                new_species.setHasOnlySubstanceUnits(species.getHasOnlySubstanceUnits())

                if species.isSetInitialConcentration():
                    new_species.setInitialConcentration(species.getInitialConcentration())
                elif species.isSetInitialAmount():
                    new_species.setInitialAmount(species.getInitialAmount())

                added_species.add(species_id)

    # Merge parameters from both models
    for model in [model1, model2]:
        for i in range(model.getNumParameters()):
            param = model.getParameter(i)
            param_id = param.getId()
            if param_id not in added_parameters:
                new_param = merged_model.createParameter()
                new_param.setId(param_id)
                new_param.setConstant(param.getConstant())
                if param.isSetValue():
                    new_param.setValue(param.getValue())
                added_parameters.add(param_id)

    # Merge reactions from both models
    for idx, model in enumerate([model1, model2]):
        for i in range(model.getNumReactions()):
            reaction = model.getReaction(i)
            reaction_id = reaction.getId()
            if reaction_id not in added_reactions:
                new_reaction = merged_model.createReaction()
                new_reaction.setId(reaction_id)
                new_reaction.setReversible(reaction.getReversible())
                new_reaction.setFast(reaction.getFast())

                # Copy reactants
                for j in range(reaction.getNumReactants()):
                    reactant = reaction.getReactant(j)
                    new_reactant = new_reaction.createReactant()
                    new_reactant.setSpecies(reactant.getSpecies())
                    new_reactant.setStoichiometry(reactant.getStoichiometry())
                    new_reactant.setConstant(reactant.getConstant())

                # Copy products
                for j in range(reaction.getNumProducts()):
                    product = reaction.getProduct(j)
                    new_product = new_reaction.createProduct()
                    new_product.setSpecies(product.getSpecies())
                    new_product.setStoichiometry(product.getStoichiometry())
                    new_product.setConstant(product.getConstant())

                # Copy modifiers
                for j in range(reaction.getNumModifiers()):
                    modifier = reaction.getModifier(j)
                    new_modifier = new_reaction.createModifier()
                    new_modifier.setSpecies(modifier.getSpecies())
                
                # Copy kinetic law
                kinetic_law = reaction.getKineticLaw()
                cloned_kinetc_law = kinetic_law.clone() if kinetic_law is not None else None
                new_reaction.setKineticLaw(cloned_kinetc_law)

                added_reactions.add(reaction_id)
    # Construct the model string
    writer = libsbml.SBMLWriter()
    sbml_str = writer.writeSBMLToString(merged_doc)
    rr = te.loadSBMLModel(sbml_str)
    #
    return rr.getAntimony()