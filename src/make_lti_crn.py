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
from typing import Optional, Tuple


def makeLtiCrn(
    num_species: int,
    num_reaction: int,
    num_products_bounds: Tuple[int, int] = (1, 3),
    kinetic_constant_bounds: Tuple[float, float] = (0.1, 10.0),
    stoichiometry_bounds: Tuple[int, int] = (1, 3),
    rate_constant_prefix: str = "k",
    species_prefix: str = "S",
    species_suffix: str = "_",
    is_input_boundary: bool = True,
    seed: Optional[int] = None
):
    """
    Generate a random linear time-invariant (LTI) chemical reaction network
    in Antimony language.
    S1 is the boundary species and its initial value is 1.
    
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
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    str : Antimony model string
    """
    ##
    def makeSpeciesName(idx: int) -> str:
        return f"{species_prefix}{idx}{species_suffix}"
    def extractSpeciesNumber(species_name: str) -> int:
        return int(species_name[len(species_prefix):-len(species_suffix)])
    ##
    input_species_name = makeSpeciesName(1)
    #
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if num_reaction < 1:
        raise ValueError("Must have at least 1 reaction")
    
    antimony_lines = []
    antimony_lines.append("# Random Chemical Reaction Network")
    antimony_lines.append("# Generated with specified constraints\n")
    antimony_lines.append("model random_crn()\n")
    
    # Track existing species and rate constants
    rate_constants = []
    
    """ # S1 is a boundary species and k1 is its initial value
    k1 = np.random.uniform(kinetic_constant_bounds[0], kinetic_constant_bounds[1])
    rate_constants.append((f"{rate_constant_prefix}1", k1))
    if is_input_boundary:
        prefix = "$"
    else:
        prefix = ""
    antimony_lines.append(f"  -> {prefix}{input_species_name}; {rate_constant_prefix}1")
    """
    antimony_lines.append(f"  species {input_species_name};\n")
    existing_species = [makeSpeciesName(n) for n in range(1, num_species + 1)]
    candidate_product_species = existing_species[2:]
    
    # Generate subsequent reactions
    reactants = []
    for rxn_idx in range(2, num_reaction + 1):
        
        # Pick a random reactant from existing species
        reactant: str = random.choice(existing_species)
        reactants.append(reactant)
        
        # Determine number of products
        num_products = random.randint(num_products_bounds[0], num_products_bounds[1])
        
        # Generate products
        products = []
        for _ in range(num_products):
            product_species = random.choice(list(candidate_product_species))
            # Assign stoichiometry
            stoich = random.randint(stoichiometry_bounds[0], stoichiometry_bounds[1])
            products.append((stoich, product_species))
        if len(products) == 0:
            continue
        
        # Generate kinetic constant
        k = np.random.uniform(kinetic_constant_bounds[0], kinetic_constant_bounds[1])
        k_name = f"{rate_constant_prefix}{rxn_idx}"
        rate_constants.append((k_name, k))
        
        # Build reaction string with spaces between stoichiometry and species
        product_str = " + ".join([f"{s} {sp}" if s > 1 else sp for s, sp in products])
        rate_law = f"{k_name} * {reactant}"
        antimony_lines.append(f"  {reactant} -> {product_str}; {rate_law}")

    # Ensure that S1 is a reactant in at least one reaction
    if input_species_name not in reactants:
        reaction_str = antimony_lines[-1]
        current_species_name = reaction_str.split(" -> ")[0].strip()
        antimony_lines[-1] = reaction_str.replace(current_species_name, input_species_name)
    
    # Add rate constant definitions
    antimony_lines.append("\n  # Rate constants")
    for k_name, k_value in rate_constants:
        antimony_lines.append(f"  {k_name} = {k_value:.4f}")
    
    # Add species initialization
    antimony_lines.append("\n  # Species initialization")
    for species in sorted(existing_species, key=lambda x: extractSpeciesNumber(x)):
        if species == input_species_name:
            if is_input_boundary:
                prefix = "$"
            else:
                prefix = ""
            antimony_lines.append(f"  {prefix}{species} = 1  # Input boundary species")
        else:
            antimony_lines.append(f"  {species} = 0")
    
    antimony_lines.append("\nend")
    antimony_str = "\n".join(antimony_lines)

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
            diag = abs(jacobian_arr[i, i])
            off_diag_sum = np.sum(np.abs(jacobian_arr[i, :])) - diag
            margin = off_diag_sum + jacobian_arr[i, i]
            if margin >= 0:
                degradation_dct[i] = 1.1*margin
        # Add degradation reactions for each species
        degradation_lines = []
        for idx, degradation_rate in degradation_dct.items():
            species = species_names[idx]
            degradation_lines.append(f"  {species} -> ; {rate_constant_prefix}d_{idx} * {species}")
            degradation_lines.append(f"  {rate_constant_prefix}d_{idx} = {degradation_rate:.4f}")
        # Insert degradation reactions before 'end'
        antimony_lines = antimony_str.splitlines()
        antimony_lines.insert(-1, "\n  # Degradation reactions")
        antimony_lines[-1:-1] = degradation_lines
        antimony_str = "\n".join(antimony_lines)

    # Check that all eigenvalues have negative real parts
    rr = te.loada(antimony_str)
    jacobian1_arr = rr.getFullJacobian()
    eigenvalues = np.linalg.eigvals(jacobian1_arr)
    if np.any(eigenvalues.real > 0):
        import pdb; pdb.set_trace()
        pass

    return antimony_str