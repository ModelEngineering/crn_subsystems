from src.merge_models import mergeModels  # type: ignore
from tests.common_test import MODEL1, MODEL2, MODEL3, MODEL_SEQUENTIAL  # type: ignore

import unittest
import libsbml # type: ignore
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict, Optional

IGNORE_TEST = False

SIMPLE_MODEL1 = """
model simple1()
    species S1, S2
    J1: S1 -> S2; k1*S1
    J2: S2 -> ; k2*S2
    $S1 = 10
    S2 = 0
    k1 = 1
    k2 = 2
    end
"""
SIMPLE_MODEL2 = """
model simple2()
    species S1, SS3
    JJ1: S1 -> SS3; kk1*S1
    JJ2: SS3 -> ; kk2*SS3
    S1 = 10
    SS3 = 0
    kk1 = 1
    kk2 = 2
    end
"""


class TestMergeModels(unittest.TestCase):

    def setUp(self):
        pass

    def testBasic(self):
        if IGNORE_TEST:
            return

        # Load models and get SBML documents
        rr1 = te.loadAntimonyModel(SIMPLE_MODEL1)
        rr2 = te.loadAntimonyModel(SIMPLE_MODEL2)
        reader = libsbml.SBMLReader()
        doc1 = reader.readSBMLFromString(rr1.getSBML())
        doc2 = reader.readSBMLFromString(rr2.getSBML())
        # Merge documents
        merged_doc = mergeModels(doc1, doc2)
        merged_model = merged_doc.getModel()
        # Check species
        expected_species_ids = {"S1", "S2", "SS3"}
        actual_species_ids = [merged_model.getSpecies(i).getId() for i in range(merged_model.getNumSpecies())]
        trues = [sid in expected_species_ids for sid in actual_species_ids]
        self.assertTrue(all(trues))
        # Check reactions
        expected_reaction_ids = {"J1", "J2", "JJ1", "JJ2"}
        actual_reaction_ids = [merged_model.getReaction(i).getId() for i in range(merged_model.getNumReactions())]
        trues = [rid in expected_reaction_ids for rid in actual_reaction_ids]
        self.assertTrue(all(trues))
        # Check parameters
        expected_parameter_ids = {"k1", "k2", "kk1", "kk2"}
        actual_parameter_ids = [merged_model.getParameter(i).getId() for i in range(merged_model.getNumParameters())]
        trues = [pid in expected_parameter_ids for pid in actual_parameter_ids]
        self.assertTrue(all(trues))

        # # Print results
        # writer = libsbml.SBMLWriter()
        # print("Merged SBML:")
        # sbml_str = writer.writeSBMLToString(merged_doc)
        # import pdb; pdb.set_trace()
        # rr = te.loadSBMLModel(sbml_str)
        # print(rr.getAntimony())
        # import pdb; pdb.set_trace()

        # # Load into tellurium and show antimony
        # merged_rr = te.loadSBMLModel(writer.writeSBMLToString(merged_doc))
        # print("\nMerged Antimony:")
        # print(merged_rr.getAntimony())





if __name__ == '__main__':
    unittest.main()