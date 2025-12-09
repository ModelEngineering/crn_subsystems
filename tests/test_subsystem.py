from src.jacobian import Jacobian
from src.model import Model  # type: ignore
from src.subsystem import Subsystem  # type: ignore
from tests.common_test import MODEL1, MODEL2, MODEL3, MODEL_SEQUENTIAL  # type: ignore

import unittest
import tellurium as te  # type: ignore
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict, Optional

IGNORE_TEST = False


class TestSubsyste(unittest.TestCase):

    def setUp(self):
        self.subsystem = Subsystem(MODEL1)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsInstance(self.subsystem.model, Model)
        self.assertIsInstance(self.subsystem.jacobian, Jacobian)
        for name in ["S1", "S2", "S3"]:
            self.assertIn(name, self.subsystem.model.species_names)
        for name in ["J1", "J2", "J3", "J4", "J5"]:
            self.assertIn(name, self.subsystem.model.reaction_names)

    def testCalculateEigenvalues(self):
        if IGNORE_TEST:
            return
        subsystem = Subsystem(MODEL_SEQUENTIAL)
        eigenvalues = subsystem.calculateEigenvalues()
        self.assertIsInstance(eigenvalues, np.ndarray)
        self.assertEqual(len(eigenvalues), 4)
        trues = [any( np.isclose(e, v) for v in [-1., -2., -3., -4.]) for e in eigenvalues]
        self.assertTrue(all(trues))

    def testCalculateSymbolicDiscs(self):
        if IGNORE_TEST:
            return
        subsystem = Subsystem(MODEL_SEQUENTIAL)
        disc_smat = subsystem.calculateSymbolicDiscs()
        self.assertIsInstance(disc_smat, sp.Matrix)
        self.assertEqual(disc_smat.shape, (subsystem.model.num_species, 2))
        self.assertEqual(disc_smat.shape, (subsystem.model.num_species, 2))
        k1, k2, k3 = sp.symbols('k1 k2 k3')
        expected_width = 2.0*sp.Matrix(
                [[0, sp.Abs(k1), sp.Abs(k2), sp.Abs(k3)]])
        width = sp.simplify(disc_smat[:, 1] - disc_smat[:, 0]) # type: ignore
        self.assertTrue(width.T == expected_width)
    
    def testCalculateNumericDiscs(self):
        if IGNORE_TEST:
            return
        subsystem = Subsystem(MODEL_SEQUENTIAL)
        disc_mat = subsystem.calculateNumericDiscs()
        disc_smat = subsystem.calculateSymbolicDiscs()
        expected_disc_mat = disc_smat.subs(subsystem.model.kinetic_constant_dct)
        self.assertTrue(np.allclose(
            disc_mat.astype(np.float64),
            np.array(expected_disc_mat).astype(np.float64)
        ))


if __name__ == '__main__':
    unittest.main()