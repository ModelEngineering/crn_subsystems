from src.merge_models import mergeModels  # type: ignore
from tests.common_test import MODEL1, MODEL2, MODEL3, MODEL_SEQUENTIAL  # type: ignore

import unittest
import pandas as pd  # type: ignore
import libsbml # type: ignore
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict, Optional

IGNORE_TEST = True
IS_PLOT = False

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
MODEL_RANDOM1 = """
model random_crn()

    # Species
    species S1_;
    species S2_;
    species S3_;
    species S4_;
    species S5_;
    J1:-> 2 S5_ + 2 S5_; k1 * S2_
    J2: S3_ -> 3 S5_; k2 * S3_
    J3: S3_ -> 3 S3_; k3 * S3_
    J4: S5_ -> S3_; k4 * S5_
    J5: S5_ -> 2 S4_ + S5_; k5 * S5_
    J6: S2_ -> 2 S4_ + 3 S4_; k6 * S2_
    J7: S4_ -> 2 S4_; k7 * S4_
    J8: S1_ -> S5_ + S5_; k8 * S1_
    J9: S1_ -> S3_ + 3 S4_; k9 * S1_
    J10: S3_ -> 2 S5_ + 2 S3_; k10 * S3_

    # Rate constants
    k1 = 0.4997
    k2 = 0.5912
    k3 = 0.4166
    k4 = 0.8616
    k5 = 0.4069
    k6 = 0.4234
    k7 = 0.2043
    k8 = 0.6065
    k9 = 0.4619
    k10 = 0.2959

    # Species initialization
    $S1_ = 1  # Input boundary species
    S2_ = 0
    S3_ = 0
    S4_ = 0
    S5_ = 0


    # Degradation reactions
    JD1: S3_ -> ; kd_1 * S3_
    kd_1 = 1.5395
    JD2: S4_ -> ; kd_2 * S4_
    kd_2 = 3.4486
    JD3: S5_ -> ; kd_3 * S5_
    kd_3 = 3.8529
    end
"""

MODEL_RANDOM2 = """
model random_crn()

    # SSpecies
    species SS10_;
    species SS11_;
    species SS12_;
    species SS13_;
    species SS14_;

    JJ1: SS11_ -> SS13_ + 2 SS12_; kk1 * SS11_
    JJ2: SS13_ -> SS14_; kk2 * SS13_
    JJ3: SS12_ -> 2 SS14_ + 2 SS12_; kk3 * SS12_
    JJ4: SS14_ -> 2 SS14_ + 3 SS14_; kk4 * SS14_
    JJ5: SS14_ -> 2 SS12_; kk5 * SS14_
    JJ6: SS10_ -> SS12_; kk6 * SS10_
    JJ7: SS12_ -> 2 SS12_; kk7 * SS12_
    JJ8: SS14_ -> 3 SS13_ + 3 SS13_; kk8 * SS14_
    JJ9: S1_ -> SS14_; kk9 * S1_
    JJ10: SS13_ -> 3 SS13_ + SS13_; kk10 * SS13_
    # Rate constants
    kk1 = 0.6300
    kk2 = 0.7799
    kk3 = 0.6599
    kk4 = 0.4522
    kk5 = 0.7942
    kk6 = 0.9919
    kk7 = 0.7141
    kk8 = 0.4096
    kk9 = 0.2442
    kk10 = 0.4076

    # SSpecies initialization
    $S1_ = 1  # Input boundary species
    SS10_ = 0
    SS11_ = 0
    SS12_ = 0
    SS13_ = 0
    SS14_ = 0


    # Degradation reactions
    JJD2: SS12_ -> ; kkd_2 * SS12_
    kkd_2 = 5.7357
    JJD3: SS13_ -> ; kkd_3 * SS13_
    kkd_3 = 3.8836
    JD4: SS14_ -> ; kkd_4 * SS14_
    kkd_4 = 2.9752
    end
"""


class TestMergeModels(unittest.TestCase):

    def setUp(self):
        pass

    def testBasic(self):
        if IGNORE_TEST:
            return

        # Load models and get SBML documents
        reader = libsbml.SBMLReader()
        # Merge documents
        merged_model_str = mergeModels(SIMPLE_MODEL1, SIMPLE_MODEL2)
        rr = te.loada(merged_model_str)
        merged_model_sbml = rr.getSBML()
        merged_model = reader.readSBMLFromString(merged_model_sbml).getModel()
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

    def testBug1(self):
        #if IGNORE_TEST:
        #    return

        # Load models and get SBML documents
        reader = libsbml.SBMLReader()
        # Merge documents
        merged_model_str = mergeModels(MODEL_RANDOM1, MODEL_RANDOM2)
        # Test valid model
        rr = te.loada(merged_model_str)
        rr.simulate(0, 10, 100)
        if IS_PLOT:
            rr.plot()
        # Compare with original models
        rr_dct = {"merged": rr}
        for i, model_str in enumerate([MODEL_RANDOM1, MODEL_RANDOM2]):
            rr_sub = te.loada(model_str)
            rr_sub.simulate(0, 10, 100)
            if IS_PLOT:
                rr_sub.plot()
            rr_dct[f"submodel_{i+1}"] = rr_sub
        # Compare the results
        dcts = {}
        for model, rr_model in rr_dct.items():
            dcts[model] = pd.Series(
                {k: v for k, v in
                    zip(rr_model.getFloatingSpeciesIds(),
                    rr_model.getSteadyStateValues())    })
        for model in [MODEL_RANDOM1, MODEL_RANDOM2]:
            rr_sub = te.loada(model)
            key = "submodel_1" if model == MODEL_RANDOM1 else "submodel_2"
            for species in rr_sub.getFloatingSpeciesIds():
                self.assertIn(species, dcts["merged"].index)
                self.assertTrue(np.isclose(
                        dcts["merged"][species], dcts[key][species], atol=1e-6))



if __name__ == '__main__':
    unittest.main()