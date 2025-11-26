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




if __name__ == '__main__':
    unittest.main()