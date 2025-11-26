import numpy as np
import pandas as pd
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
