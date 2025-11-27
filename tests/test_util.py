import src.util as util

import unittest
import numpy as np
import sympy as sp  # type: ignore
from typing import Dict

IGNORE_TEST = False


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

    def testSolveLinearSystem(self):
        if IGNORE_TEST:
            return
        A = np.array([[1, 2, 3],
                    [0, 1, 4],
                    [5, 6, 0]], dtype=float)
        b = np.array([14, 13, 32], dtype=float)
        fixed = {0: 2}  # Fix x0 = 2
        x, residual, rank = util.solveLinearSystem(A, b, fixed)
        expected = np.array([2, 3, 1], dtype=float)
        self.assertTrue(np.allclose(x, expected))




if __name__ == '__main__':
    unittest.main()