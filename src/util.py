import numpy as np
import pandas as pd # type: ignore
import sympy as sp  # type: ignore
from typing import List, Optional, Any


def mat2DF(mat, column_names: Optional[List[str]]=None, row_names: Optional[List[str]]=None)->pd.DataFrame:
    """
    Converts a numpy ndarray or array-like to a DataFrame.

    Parameters
    ----------
    mat: np.Array, NamedArray, DataFrame
    column_names: list-str
    row_names: list-str
    """
    if isinstance(mat, pd.DataFrame):
        df = mat
    else:
        if len(np.shape(mat)) == 1:
            mat = np.reshape(mat, (len(mat), 1))
        if column_names is None:
            if hasattr(mat, "colnames"):
                column_names = mat.colnames # type: ignore
        if column_names is not None:
            if len(column_names) == 0:
                column_names = None
        if row_names is None:
            if hasattr(mat, "rownames"):
                if len(mat.rownames) > 0:  # type: ignore
                    row_names = mat.rownames  # type: ignore
        if row_names is not None:
            if len(row_names) == 0:
                row_names = None
        df = pd.DataFrame(mat, columns=column_names, index=row_names)
    return df

def subsUsingName(expr: Any, subs_dct: dict)->Any:
    """
    Substitute values into a numpy ndarray of sympy expressions using a substitution dictionary.

    Parameters
    ----------
    expr: np.ndarray
        Numpy array of sympy expressions
    subs_dct: dict
        Dictionary mapping strings to values

    Returns
    -------
    Any
    """
    symbols = expr.free_symbols
    str_to_symbol = {str(s): s for s in symbols}
    actual_subs_dct = {str_to_symbol[k]: v for k, v in subs_dct.items() if k in str_to_symbol}
    return expr.subs(actual_subs_dct)

def solveLinearSystem(A, b, fixed=None):
    """
    Solve Ax = b in the least-squares sense, with optional fixed coordinates.
    
    Args:
        A: matrix (m x n)
        b: right-hand side vector (length m)
        fixed: dict mapping coordinate index -> fixed value
            e.g., {0: 1.5, 3: 0.0} fixes x[0]=1.5 and x[3]=0.0
    
    Returns:
        x: solution
        residual: norm of (Ax - b)
        rank: effective rank of the (reduced) system
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]
    # Handle case with no fixed coordinates
    if fixed is None:
        fixed = {}
    # Identify free vs fixed indices
    fixed_indices = sorted(fixed.keys())
    free_indices = [i for i in range(n) if i not in fixed]
    if not free_indices:
        # All coordinates fixed; just compute residual
        x = np.array([fixed[i] for i in range(n)])
        residual = np.linalg.norm(A @ x - b)
        return x, residual, 0
    # Move fixed terms to the right-hand side: A_free @ x_free = b - A_fixed @ x_fixed
    A_free = A[:, free_indices]
    A_fixed = A[:, fixed_indices]
    x_fixed = np.array([fixed[i] for i in fixed_indices])
    #x_fixed = np.reshape(x_fixed, (-1, 1))
    b_adjusted = b - (A_fixed @ x_fixed)
    # Solve reduced system
    result = np.linalg.lstsq(A_free, b_adjusted, rcond=None)
    x_free_solutions = [result[0].flatten()]
    rank = result[2]
    
    # If system is underdetermined, there are infinitely many solutions
    # We'll sample a few and pick the one with smallest residual
    if rank < len(free_indices):
        # Get the null space of A_free
        _, s, Vt = np.linalg.svd(A_free, full_matrices=True)
        # Null space corresponds to singular values that are effectively zero
        #tol = s[0] * max(A_free.shape) * np.finfo(float).eps if len(s) > 0 else 0
        null_space = Vt[rank:, :].T  # columns are basis vectors for null space
        
        # Generate additional candidate solutions by adding null space components
        # Try a few different linear combinations
        for scale in [-10, -1, -0.1, 0.1, 1, 10]:
            for i in range(null_space.shape[1]):
                x_candidate = x_free_solutions[0] + scale * null_space[:, i]
                x_free_solutions.append(x_candidate)
    
    # Evaluate residual for each candidate solution and pick the best
    best_residual = float('inf')
    best_x = None
    
    for x_free in x_free_solutions:
        x = np.zeros(n)
        for i, idx in enumerate(fixed_indices):
            x[idx] = fixed[idx]
        for i, idx in enumerate(free_indices):
            x[idx] = x_free[i]
        residual = np.linalg.norm(A @ x - b)
        if residual < best_residual:
            best_residual = residual
            best_x = x
    
    return best_x, best_residual, rank