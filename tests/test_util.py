import src.util as util

import unittest
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict

IGNORE_TEST = False
IS_PLOT = False


class TestFunctions(unittest.TestCase):

    def testSubsUsingName(self):
        if IGNORE_TEST:
            return
        x, y, z = sp.symbols('x y z')
        expr = sp.Matrix([[x + y], [y + z], [z + x]])
        subs_dct = {'x': 1, 'y': 2}
        result = util.subsUsingName(expr, subs_dct)
        expected = sp.Matrix([[1 + 2], [2 + z], [z + 1]])
        self.assertTrue(result.equals(expected))

    def testSolveLinearSystemSingleSolution(self):
        if IGNORE_TEST:
            return
        A = np.array([[0, 0, 0],
                    [1, -2, 0],
                    [0, 2, -3]], dtype=float)
        b = np.array([1, 0, 0], dtype=float)
        fixed = {0:1}
        x, residual, rank = util.solveLinearSystem(A, b, fixed)
        expected = np.array([1, 0.5, 0.333], dtype=float)
        self.assertTrue(np.allclose(x, expected, atol=1e-2))  # type: ignore

    # FIXME: Poor test for multiple solutions case 
    def testSolveLinearSystemMultipleSolution(self):
        if IGNORE_TEST:
            return
        A = np.array([[0, 0, 0],
                    [1, -2, 0],
                    [1, -2, 0]], dtype=float)
        b = np.array([1, 0, 0], dtype=float)
        fixed = {0:1}
        x, residual, rank = util.solveLinearSystem(A, b, fixed)
        expected = np.array([1, 0.5, 0], dtype=float)
        self.assertTrue(np.allclose(x, expected, atol=1e-2))  # type: ignore




if __name__ == '__main__':
    unittest.main()