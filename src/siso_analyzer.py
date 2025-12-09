'''Does SISO analysis of an LTI Antimony model.'''

"""
Conventions for Antimony models.
1. All reactions are either bounderies or uni-uni reactions.
2. Species are named S{num}, where num is the species index starting at 1. S1 is a boundary species.
"""


from src.model import Model  # type: ignore
from src.jacobian import Jacobian  # type: ignore
from src.subsystem import Subsystem  # type: ignore

from collections import namedtuple
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd # type: ignore
from scipy import signal  # type: ignore
import sympy as sp  # type: ignore
import tellurium as te # type: ignore
from typing import Tuple, Union, Optional, cast, Any

from src.util import solveLinearSystem

# Constants
# Null values for key attributes
NULL_TRANSFER_FUNCTION_EXPR = sp.Symbol("Unknown")
NULL_TRANSFER_FUNCTION_MATRIX = sp.Matrix()
NULL_SPECIES_NAME = ""
DEFAULT_INPUT_NAME = "S1_"


MakeSymbolicAMatResult = namedtuple("MakeSymbolicAMatResult", ["A_sym", "rate_dct"])


class SISOAnalyzer(object):
    def __init__(self, antimony_str: str,
            input_name:str = DEFAULT_INPUT_NAME,
            output_name:str = NULL_SPECIES_NAME,
            system_name: str = "") -> None:
        """
        Initializes the NetworkAnalyzer with an optional Antimony model string.

        Parameters:
        -----------
        antimony_str: Antimony model string
            Antimony model string to initialize the NetworkAnalyzer
        input_name: Species name to use as input (default: S1)
        output_name: Species name to use as output (default: largest numbered species)
        system_name: Name of the system (default: "")

        Returns:
        --------
        None
        """
        # FIXME: Use Subsystem class
        # Initialize to the correct type. Changing self.antimony_str triggers
        # recalculation of jacobian matrices and kinetic_constant_dct
        self.original_antimony_str = antimony_str
        self.antimony_str = antimony_str.replace(f"${input_name}", f"{input_name}")  # Remove boundary marker for Tellurium
        self.model = Model(self.antimony_str)
        self.jacobian = Jacobian(self.model)
        self.original_model = Model(self.original_antimony_str)
        self.input_name = input_name
        # Initialize other properties
        self.jacobian = Jacobian(self.model)
        self.jacobian_df = self.jacobian.jacobian_df
        self.jacobian_smat = self.jacobian.jacobian_smat
        self.b_smat = self.jacobian.b_smat
        self.kinetic_constant_dct = self.model.kinetic_constant_dct
        self.species_names = self.model.species_names
        self.num_species = self.model.num_species
        self.output_name = self._updateOutputName(output_name)
        self.input_species_index = self.model.getSpeciesIndex(self.input_name)
        self.output_species_index = self.model.getSpeciesIndex(self.output_name)
        # The following are calculaed via deferred evaluation after initialize
        self._transfer_function_smat = NULL_TRANSFER_FUNCTION_MATRIX
        self._transfer_function_expr = NULL_TRANSFER_FUNCTION_EXPR

    def _updateOutputName(self, proposed_output_name) -> str:
        """
        Updates the output species name.

        Returns:
        --------
            output_name
        """
        candidate_names = self.model.roadrunner.getExtendedStoichiometryMatrix().rownames
        if not self.input_name in candidate_names:
            raise ValueError(f"Input species {self.input_name} not found in model species: {candidate_names}")
        if proposed_output_name == NULL_SPECIES_NAME:
            candidate_names = cast(list[str], self.jacobian_df.columns)  # type: ignore
            output_name = candidate_names[-1]
        else:
            output_name = proposed_output_name
        if output_name not in candidate_names:
            import pdb; pdb.set_trace()
            raise ValueError(f"Output species {self.output_name} not found in model species: {candidate_names}")
        #
        return output_name

########### GETTERS AND SETTERS #############
    @property
    def transfer_function_smat(self) -> sp.Matrix:
        """Returns the symbolic transfer function expression."""
        if self._transfer_function_expr == NULL_TRANSFER_FUNCTION_EXPR:
            # Construct the transfer function expression
            input_index = self.species_names.index(self.input_name)
            s = sp.Symbol("s")
            B = sp.eye(self.jacobian_smat.rows)
            I = sp.eye(self.jacobian_smat.rows)
            A_smat = self.jacobian_smat
            b_smat = sp.Matrix.zeros(self.jacobian_smat.rows, 1)
            b_smat[input_index] = 1
            b_smat += self.b_smat
            A_smat[input_index, :] = sp.Matrix.zeros(1, self.jacobian_smat.cols)
            # Multiply the step response by s to get the transfer function
            self._transfer_function_smat = sp.simplify(s*((s*I - A_smat).inv() * B * b_smat))
        return self._transfer_function_smat
    
    @property
    def transfer_function_expr(self)->Any:
        """Returns the symbolic transfer function expression."""
        if self._transfer_function_expr == NULL_TRANSFER_FUNCTION_EXPR:
            # Construct the transfer function expression
            input_index = self.species_names.index(self.input_name)
            output_index = self.species_names.index(self.output_name)
            self._transfer_function_expr = self.transfer_function_smat[output_index, input_index]
        return self._transfer_function_expr
    

    ############ METHODS #############

    def makeSequentialAntimony(self, num_species) -> str:
        """
        Generates an Antimony file consisting of num_stage sequences of uni-uni reactions 
        and a degradation reaction. The kinetic constants have sequential values and are 
        labelled "k1", "k2", "k3", etc. with values 1, 2, 3, etc.
        
        Args:
            num_stage: Number of stages in the sequential network
            
        Returns:
            str: Antimony model string
            
        Example:
            For num_stage=3:
            - S0 -> S1; k1*S0   (k1 = 1)
            - S1 -> ; k2*S1     (k2 = 2)
            - S1 -> S2; k3*S1   (k3 = 3)
            - S2 -> ; k4*S2     (k4 = 4)
            - S2 -> S3; k5*S2   (k5 = 5)
            - S3 -> ; k6*S3     (k6 = 6)
        """

        lines = []
        lines.append("model *sequential_network()")
        
        # Generate reactions and parameters for each stage
        k_counter = 1
        for i in range(num_species):
            source_species = f"S{i}"
            target_species = f"S{i+1}"
            
            # Uni-uni reaction: Si -> Si+1
            forward_rate_name = f"k{k_counter}"
            lines.append(f"    {source_species} -> {target_species}; {forward_rate_name}*{source_species}")
            k_counter += 1
            
            # Degradation reaction: Si -> ;
            deg_rate_name = f"k{k_counter}"
            lines.append(f"    {source_species} -> ; {deg_rate_name}*{source_species}")
            k_counter += 1
        
        lines.append("")
        
        # Define kinetic constants
        for i in range(1, k_counter):
            lines.append(f"    k{i} = {i}")
        
        lines.append("")
        
        # Initialize species concentrations
        for i in range(num_species + 1):
            lines.append(f"    S{i} = 0")
        
        lines.append("end")
        
        return "\n".join(lines)

    def makeTransferFunction(self, **kwargs) -> signal.TransferFunction:
        """
        Uses the current symbolic A matrix to construct a transfer function from S1 to
        the largest numbered species on output.
        
        Parameters:
        -----------
        kwargs: name-value pairs for parameters

        State updates
        ----------
        self.kinetic_constant_dct: Dictionary of kinetic constants
        self.transfer_function_expr: Symbolic transfer function expression
        self.transfer_function: sign.TransferFunction object

        Returns:
        --------
        signal.TransferFunction
            Transfer function object
        """
        kinetic_constant_dct = dict(self.kinetic_constant_dct)
        kinetic_constant_dct.update(kwargs)
        # Substitute kinetic constants into symbolic transfer function expression
        transfer_function_smat = sp.simplify(self.transfer_function_expr.subs(kinetic_constant_dct))
        # All free symbols except for s should be removed. Now convert to transfer function
        free_syms = transfer_function_smat.free_symbols
        if len(free_syms) == 0:
            return signal.TransferFunction([float(transfer_function_smat)], [1])
        elif len(free_syms) > 1:
            raise ValueError(f"Expression contains multiple symbols: {free_syms}. "
                            "Only single-variable rational functions are supported.")
        s = free_syms.pop()
        numer, denom = sp.fraction(transfer_function_smat)
        # Convert to polynomials
        numer_poly = sp.Poly(numer, s)
        denom_poly = sp.Poly(denom, s)
        # Extract coefficients (from highest to lowest degree)
        numer_coeffs = [float(c) for c in numer_poly.all_coeffs()]
        denom_coeffs = [float(c) for c in denom_poly.all_coeffs()]
        # Create transfer function
        self.transfer_function = signal.TransferFunction(numer_coeffs, denom_coeffs) 
        #
        return self.transfer_function

    def plotTransferFunctionValidation(self, step_size: float = 1.0) -> None:
        """
        Plots the simulated and transfer function predicted outputs.

        Args:
            step_size (float): The magnitude of the step input.
        """
        tf = self.makeTransferFunction()
        rr = te.loada(self.original_antimony_str)
        data = rr.simulate(0, 10, 100)
        times = data[:, 0]
        _, predicted_y = signal.step(tf, T=times)
        true_y = data[f"[{self.output_name}]"]
        # Plot the step response comparison
        plt.figure()
        plt.scatter(true_y, predicted_y*step_size)
        plt.plot([0, max(true_y)], [0, max(true_y)], 'r--', label="Ideal")
        plt.title(f"Predicted vs. Actual Step Response for {self.output_name}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid()
        plt.legend(["Comparison", "Ideal"])
        plt.show()

    # TODO: delete. Use subsystem
    def calculateStepResponse(self, step_size: float = 1.0) -> float:
        """
        Calculates the step response using steady state analysis by setting all derivatives to zero.
        to zero and requiring that the input species increases by step_size.

        Args:
            time_vec (np.ndarray): Time vector for the step response.
            step_size (float): The magnitude of the step input.

        Returns:
            step response (float): ratio of output to input
        """
        b_mat = np.zeros((self.num_species))
        b_mat[self.input_species_index] = step_size
        A_mat = np.array(self.jacobian_df.values, dtype=float)
        A_mat[self.input_species_index, :] = np.zeros((1, self.num_species))
        # Solve Ax = -b for steady state
        x_ss, residual, _ = solveLinearSystem(A_mat, b_mat, 
                fixed={self.input_species_index: step_size})
        if residual > 1e-6:
            #raise ValueError(f"Could not solve for steady state: residual={residual}")
            pass
        step_response = x_ss[self.output_species_index] / step_size  # type: ignore
        return step_response