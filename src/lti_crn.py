'''Generates random chemical reaction networks in Antimony language.'''

"""
Characteristics of the generated CRN.
1. Behaves as a linear system of ODEs.
2. Each reaction has zero or one reactant and zero or more products.
3. Kinetic constants are randomly assigned within specified bounds.
4. Stoichiometry of products is randomly assigned within specified bounds.
5. The first reaction has no reactant (i.e., it is a source reaction).
6. Subsequent reactions select reactants from existing species to ensure connectivity.
7. Species are named sequentially (S1_, S2_, S3_, ...).
8. Kinetic constants are named sequentially (k1, k2, k3,
9. Species S1 is an input boundary with the kinetic constant k1 = 1.
10. The system is assured to be stable by adding degradation reactions if necessary.
11. The system is assured to be non-trivial by ensuring that S1 is a reactant in at least one reaction.
"""

import random
import numpy as np
import tellurium as te  # type: ignore
from typing import Optional, Tuple, List

class LtiCrn(object):

    def __init__(self,
        num_species: int,
        num_reaction: int,
        num_products_bounds: Tuple[int, int] = (1, 3),
        kinetic_constant_bounds: Tuple[float, float] = (0.1, 10.0),
        stoichiometry_bounds: Tuple[int, int] = (1, 3),
        rate_constant_prefix: str = "k",
        species_prefix: str = "S",
        species_suffix: str = "_",
        is_input_boundary: bool = True,
        input_species_indices:  list[int] = [1],
        starting_species_index: int = 1,
        is_test_mode: bool = False,
        seed: Optional[int] = None):
        """
        Generate a random linear time-invariant (LTI) chemical reaction network
        in Antimony language.
        Boundary species have their initial value set to 1.
        Boundary species are indicated by the boundary_species_indices parameter.
        
        Parameters:
        -----------
        num_species : int
            Maximum number of species to generate
        num_reaction : int
            Total number of reactions to generate
        num_products_bounds : tuple of (int, int)
            (min, max) number of products per reaction
        kinetic_constant_bounds : tuple of (float, float)
            (min, max) values for kinetic rate constants
        stoichiometry_bounds : tuple of (int, int)
            (min, max) stoichiometry coefficients for product species
        rate_constant_prefix : str
            Prefix for naming kinetic rate constant: Default is 'k'
        species_prefix : str
            Prefix for naming species: Default is 'S'
        species_suffix : str
            Suffix for naming species: Default is '_'
        is_input_boundary : bool
            If True, species S1 is treated as an input boundary species
        input_species_indices:  list[int]
            List of species indices to be treated as input boundary species
        starting_species_index : int
            Starting index for species naming
        is_test_mode : bool
            If True, generates a simpler model for testing purposes
        seed : int, optional
            Random seed for reproducibility
        """
        if num_reaction < 1:
            raise ValueError("Must have at least 1 reaction")
        #
        self.num_species = num_species
        self.num_reaction = num_reaction
        self.num_products_bounds = num_products_bounds
        self.kinetic_constant_bounds = kinetic_constant_bounds
        self.stoichiometry_bounds = stoichiometry_bounds
        self.rate_constant_prefix = rate_constant_prefix
        self.species_prefix = species_prefix
        self.species_suffix = species_suffix
        self.is_input_boundary = is_input_boundary
        self.input_indices = input_species_indices
        self.starting_species_index = starting_species_index
        self.seed = seed
        # Calculated attributes
        self.input_names = [self._makeSpeciesName(i) for i in self.input_indices]
        self.species_names = [self._makeSpeciesName(n) for n in range(self.starting_species_index,
                self.starting_species_index + self.num_species)]
        for input_species_name in self.input_names:
            self.species_names.append(input_species_name)
        self.species_names = list(set(self.species_names))
        #
        if is_test_mode:
            return
        #
        self.antimony_str = self.generate()

    def _makeSpeciesName(self, idx: int) -> str:
        return f"{self.species_prefix}{idx}{self.species_suffix}"
    
    def _extractSpeciesNumber(self, species_name: str) -> int:
        return int(species_name[len(self.species_prefix):-len(self.species_suffix)])

    def _addDegradationReactions(self, antimony_str: str,
            degradation_dct: dict[str, float], rate_constant_prefix: str) -> str:
        """ Adds degradation reactions to the Antimony string for the specified species.
        Degradation reactions are of the form:
            JDi: Si -> ; kd_i * Si
        """
        degradation_lines:List[str] = []
        for species_name, degradation_rate in degradation_dct.items():
            if np.isclose(degradation_rate, 0):
                continue
            rate_constant_name = f"{rate_constant_prefix}d_{species_name}"
            degradation_lines.append(f"  JD{species_name}: {species_name} -> ; {rate_constant_name} * {species_name}")
            degradation_lines.append(f"  {rate_constant_name} = {degradation_rate:.4f}")
        # Update the antimony string
        antimony_lines = antimony_str.splitlines()
        antimony_lines.insert(-1, "\n  # Degradation reactions")
        antimony_lines[-1:-1] = degradation_lines  # type: ignore
        antimony_str = "\n".join(antimony_lines)
        #
        return antimony_str
    
    def _stablizeCrn(self, antimony_str: str) -> str:
        """ Adds degradation reactions to ensure the CRN is stable.
        Uses Gershgorin's Circle Theorem to check stability.
        """
        # Make the network stable by adding degradation reactions.
        # We will use Gershgorin's Circle Theorem to check stability.
        rr = te.loada(antimony_str)
        jacobian_arr = rr.getFullJacobian()
        species_names = jacobian_arr.colnames
        eigenvalues = np.linalg.eigvals(jacobian_arr)
        if np.any(eigenvalues.real > -0.1):
            # Find the columns in the jacobian where the sum of the absolute values of the non-diagonal entries
            # in the same row is greater than or equal to the absolute value of the diagonal entry.
            degradation_dct = {}
            for i in range(jacobian_arr.shape[0]):
                #diag = abs(jacobian_arr[i, i])
                #off_diag_sum = np.sum(np.abs(jacobian_arr[i, :])) - diag
                #margin = off_diag_sum + jacobian_arr[i, i]
                species_name = species_names[i]
                diag = abs(jacobian_arr[i, i])
                margin = np.sum(np.abs(jacobian_arr[i, :])) - np.abs(diag) + diag
                if margin >= 0:
                    degradation_dct[species_name] = 1.1*margin
            # Add degradation reactions for each species
            antimony_str = self._addDegradationReactions(antimony_str,
                    degradation_dct, self.rate_constant_prefix) 
        # Add more degradations if still not bounded output
        for idx in range(5):
            rr = te.loada(antimony_str)
            jacobian1_arr = rr.getFullJacobian()
            eigenvalues = np.linalg.eigvals(jacobian1_arr)
            if np.all(eigenvalues.real < 0):
                break
            degradation_dct = {n: 10**idx for n in species_names}
            antimony_str = self._addDegradationReactions(antimony_str,
                    degradation_dct, self.rate_constant_prefix) 
        else:
            raise RuntimeError("Could not stabilize the generated CRN.")
        #
        return antimony_str

    def generate(self) -> str:
        """Generates the Antimony string for the network

        Returns:
            str
        """
        #
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        antimony_lines = []
        antimony_lines.append("# Random Chemical Reaction Network")
        antimony_lines.append("# Generated with specified constraints\n")
        antimony_lines.append("model random_crn()\n")
        # Initializations
        rate_constants = []
        candidate_product_species = [s for s in self.species_names if s not in self.input_names]
        # Define the species
        antimony_lines.append("  # Species")
        for species_name in self.species_names:
            antimony_lines.append(f"  species {species_name};")
        antimony_lines.append("")
        # Generate subsequent reactions
        reactants = []
        for rxn_idx in range(1, self.num_reaction + 1):
            # Pick a random reactant from existing species
            reactant: str = random.choice(self.species_names)
            reactants.append(reactant)
            # Determine number of products
            num_products = random.randint(self.num_products_bounds[0], self.num_products_bounds[1])
            # Generate products
            products = []
            for _ in range(num_products):
                product_species = random.choice(list(candidate_product_species))
                # Assign stoichiometry
                stoich = random.randint(self.stoichiometry_bounds[0], self.stoichiometry_bounds[1])
                products.append((stoich, product_species))
            if len(products) == 0:
                continue
            # Generate kinetic constant
            k = np.random.uniform(self.kinetic_constant_bounds[0], self.kinetic_constant_bounds[1])
            k_name = f"{self.rate_constant_prefix}{rxn_idx}"
            rate_constants.append((k_name, k))
            
            # Build reaction string with spaces between stoichiometry and species
            product_str = " + ".join([f"{s} {sp}" if s > 1 else sp for s, sp in products])
            rate_law = f"{k_name} * {reactant}"
            antimony_lines.append(f"  J{rxn_idx}: {reactant} -> {product_str}; {rate_law}")
        # Ensure that an input species is a reactant in at least one reaction
        # FIXME: May replace an existing reactant that is also an input species
        for idx, input_species_name in enumerate(self.input_names):
            if input_species_name not in reactants:
                reaction_str = antimony_lines[-(idx+1)]
                reaction_id = reaction_str.split(":")[0]
                current_species_name = reaction_str.split(" -> ")[0].strip()
                new_reaction_str = reaction_str.replace(reaction_id + current_species_name, input_species_name)
                antimony_lines[-(idx+1)] = new_reaction_str
        # Add rate constant definitions
        antimony_lines.append("\n  # Rate constants")
        for k_name, k_value in rate_constants:
            antimony_lines.append(f"  {k_name} = {k_value:.4f}")
        # Add species initialization
        antimony_lines.append("\n  # Species initialization")
        for species in sorted(self.species_names, key=lambda x: self._extractSpeciesNumber(x)):
            if species in self.input_names:
                if self.is_input_boundary:
                    prefix = "$"
                else:
                    prefix = ""
                antimony_lines.append(f"  {prefix}{species} = 1  # Input boundary species")
            else:
                antimony_lines.append(f"  {species} = 0")
        # Terminate the model so can evaluate it in Roadrunner
        antimony_lines.append("\nend")
        antimony_str = "\n".join(antimony_lines)
        # Make the network stable by adding degradation reactions.
        # We will use Gershgorin's Circle Theorem to check stability.
        return self._stablizeCrn(antimony_str)