import plip
from plip import structure
from plip.structure.preparation import PDBComplex
import rdkit
from rdkit import Chem
import pandas as pd
from typing import List, Tuple, NamedTuple
from pymol import cmd


class Interaction:
    """
    Class to represent an interaction between a protein and a ligand
    """

    def __init__(
        self,
        interaction_type: str,
        protein_atom: int | List,
        protein_chain: str,
        protein_residue: str,
        protein_number: int,
        ligand_atom: int | List,
        ligand_id: int | List,
        distance: float,
        angle: float,
        plip_id: int,
    ) -> None:
        """
        Args:
            interaction_type (str): The type of interaction.
            protein_atom (list): The protein atom.
            protein_chain (str): The protein chain.
            protein_residue (str): The protein residue.
            protein_number (int): The protein number.
            ligand_atom (list): The ligand atom.
            ligand_id (str): The ligand ID.
            distance (float): The distance between the protein and ligand atoms.
            angle (float): The angle between the protein and ligand atoms.
            plip_id (str): The PLIP ID.
        """
        self.interaction_type = interaction_type
        self.protein_atom = protein_atom
        self.protein_chain = protein_chain
        self.protein_residue = protein_residue
        self.protein_number = protein_number
        self.ligand_atom = ligand_atom
        self.ligand_id = ligand_id
        self.distance = distance
        self.angle = angle
        self.plip_id = plip_id

    def to_tuple(self):
        """
        Convert the Interaction object to a tuple.
        """
        return (
            self.interaction_type,
            self.protein_atom,
            self.protein_chain,
            self.protein_residue,
            self.protein_number,
            self.ligand_atom,
            self.ligand_id,
            self.distance,
            self.angle,
            self.plip_id,
        )


def parse_interaction(interaction: NamedTuple) -> Interaction:
    """
    Parse a PLIP interaction object into an Interaction object.

    Args:
        interaction (NamedTuple): The PLIP interaction object.

    Returns:
        Interaction (Interaction): The parsed Interaction object.
    """
    if "saltbridge" in str(type(interaction)):
        return Interaction("saltbridge", *process_saltbridge(interaction))
    elif "hydroph" in str(type(interaction)):
        return Interaction("hydrophobic", *process_hydrophobic(interaction))
    elif "hbond" in str(type(interaction)):
        return Interaction("hbond", *process_hbond(interaction))
    elif "pistack" in str(type(interaction)):
        return Interaction("pistack", *process_pi_stack(interaction))
    elif "pication" in str(type(interaction)):
        return Interaction("pication", *process_pication(interaction))
    elif "halogenbond" in str(type(interaction)):
        return Interaction("halogenbond", *process_halogenbond(interaction))
    else:
        raise NotImplementedError(f"Parsing not implemented for {type(interaction)}")


def process_pi_stack(interaction: NamedTuple) -> Tuple:
    """
    Process a pi-stacking interaction from NamedTuple to tuple.

    Args:
        interaction (NamedTuple): The pi-stacking interaction.

    Returns:
        tuple: The processed pi-stacking interaction.
    """
    protein_ring_atoms = [
        (j.coords, j.atomicnum) for j in interaction.proteinring.atoms
    ]
    protein_chain = interaction.reschain
    protein_residue = interaction.restype
    protein_number = interaction.resnr
    ligand_ring_atoms = [(j.coords, j.atomicnum) for j in interaction.ligandring.atoms]
    ligand_id = [id for id in interaction.ligandring.atoms_orig_idx]
    distance = interaction.distance
    angle = interaction.angle
    plip_id = None
    return (
        protein_ring_atoms,
        protein_chain,
        protein_residue,
        protein_number,
        ligand_ring_atoms,
        ligand_id,
        distance,
        angle,
        plip_id,
    )


def process_hydrophobic(interaction: NamedTuple) -> Tuple:
    """
    Process a hydrophobic interaction from NamedTuple to Interaction.

    Args:
        interaction (NamedTuple): The hydrophobic interaction.

    Returns:
        tuple: The processed hydrophobic interaction.
    """
    protein_atom = [(interaction.bsatom.coords, interaction.bsatom.atomicnum)]
    protein_chain = interaction.reschain
    protein_residue = interaction.restype
    protein_number = interaction.resnr
    ligand_atom = [(interaction.ligatom.coords, interaction.ligatom.atomicnum)]
    ligand_id = interaction.ligatom_orig_idx
    distance = interaction.distance
    plip_id = None
    return (
        protein_atom,
        protein_chain,
        protein_residue,
        protein_number,
        ligand_atom,
        ligand_id,
        distance,
        None,
        plip_id,
    )


def process_hbond(interaction: NamedTuple) -> Tuple:
    """
    Process a hydrogen bond interaction from NamedTuple to tuple.

    Args:
        interaction (NamedTuple): The hydrogen bond interaction.

    Returns:
        tuple: The processed hydrogen bond interaction.
    """
    if interaction.protisdon:
        protein_atom = [(interaction.d.coords, interaction.d.atomicnum)]
        ligand_atom = [(interaction.a.coords, interaction.a.atomicnum)]
        ligand_id = interaction.a_orig_idx
    else:
        protein_atom = [(interaction.a.coords, interaction.a.atomicnum)]
        ligand_atom = [(interaction.d.coords, interaction.d.atomicnum)]
        ligand_id = interaction.d_orig_idx
    protein_chain = interaction.reschain
    protein_residue = interaction.restype
    protein_number = interaction.resnr
    distance = interaction.distance_ad
    angle = interaction.angle
    plip_id = None
    return (
        protein_atom,
        protein_chain,
        protein_residue,
        protein_number,
        ligand_atom,
        ligand_id,
        distance,
        angle,
        plip_id,
    )


def process_saltbridge(interaction: NamedTuple) -> Tuple:
    """
    Process a salt bridge interaction from NamedTuple to tuple.

    Args:
        interaction (NamedTuple): The salt bridge interaction.

    Returns:
        tuple: The processed salt bridge interaction.
    """
    if interaction.protispos:
        protein_atom = [(a.coords, a.atomicnum) for a in interaction.positive.atoms]
        ligand_atom = [(a.coords, a.atomicnum) for a in interaction.negative.atoms]
        ligand_id = interaction.negative.atoms_orig_idx
    else:
        protein_atom = [(a.coords, a.atomicnum) for a in interaction.negative.atoms]
        ligand_atom = [(a.coords, a.atomicnum) for a in interaction.positive.atoms]
        ligand_id = interaction.positive.atoms_orig_idx
    protein_chain = interaction.reschain
    protein_residue = interaction.restype
    protein_number = interaction.resnr
    distance = interaction.distance
    plip_id = None
    return (
        protein_atom,
        protein_chain,
        protein_residue,
        protein_number,
        ligand_atom,
        ligand_id,
        distance,
        None,
        plip_id,
    )


def process_halogenbond(interaction: NamedTuple) -> Tuple:
    """
    Process a halogen bond interaction from NamedTuple to tuple.

    Args:
        interaction (NamedTuple): The halogen bond interaction.

    Returns:
        tuple: The processed halogen bond interaction.
    """
    # assumes ligand is the donor
    protein_atom = [
        (interaction.acc.o.coords, interaction.acc.o.atomicnum),
        (interaction.acc.y.coords, interaction.acc.y.atomicnum),
    ]
    ligand_atom = [(interaction.don.x.coords, interaction.don.x.atomicnum)]
    protein_chain = interaction.reschain
    protein_residue = interaction.restype
    protein_number = interaction.resnr
    distance = interaction.distance
    angle = [(interaction.don_angle, interaction.acc_angle)]
    ligand_id = interaction.don_orig_idx
    plip_id = None
    return (
        protein_atom,
        protein_chain,
        protein_residue,
        protein_number,
        ligand_atom,
        ligand_id,
        distance,
        angle,
        plip_id,
    )


def process_pication(interaction: NamedTuple) -> Tuple:
    """
    Process a pi-cation interaction from NamedTuple to tuple.

    Args:
        interaction (NamedTuple): The pi-cation interaction.

    Returns:
        tuple: The processed pi-cation interaction.
    """
    if interaction.protcharged:
        protein_atom = [(a.coords, a.atomicnum) for a in interaction.charge.atoms]
        ligand_atom = [(a.coords, a.atomicnum) for a in interaction.ring.atoms]
        ligand_id = interaction.ring.atoms_orig_idx
    else:
        protein_atom = [(a.coords, a.atomicnum) for a in interaction.ring.atoms]
        ligand_atom = [(a.coords, a.atomicnum) for a in interaction.charge.atoms]
        ligand_id = interaction.charge.atoms_orig_idx
    protein_chain = interaction.reschain
    protein_residue = interaction.restype
    protein_number = interaction.resnr
    distance = interaction.distance
    plip_id = None
    return (
        protein_atom,
        protein_chain,
        protein_residue,
        protein_number,
        ligand_atom,
        ligand_id,
        distance,
        None,
        plip_id,
    )


def _interactions_to_dataframe(interaction_list: list[Interaction]) -> pd.DataFrame:
    """
    Convert a list of Interaction objects to a DataFrame.

    Args:
        interaction_list (list): A list of Interaction objects.

    Returns:
        pd.DataFrame: The interactions as a DataFrame.
    """
    columns = [
        "type",
        "protein_atom",
        "protein_chain",
        "protein_residue",
        "protein_number",
        "ligand_atom",
        "ligand_id",
        "distance",
        "angle",
        "plip_id",
    ]

    interactions_as_tuples = [
        interaction.to_tuple() for interaction in interaction_list
    ]
    interactions = list(zip(*interactions_as_tuples))
    if len(interactions) > 0:
        interactions_as_dict = {
            columns[i]: interactions[i] for i in range(len(columns))
        }
        return pd.DataFrame(interactions_as_dict)
    else:
        return pd.DataFrame(columns=columns)


def get_interactions(plip_obj: PDBComplex, ligand_code: str = "UNK") -> pd.DataFrame:
    """
    Process the interactions between a protein and a ligand and return them as a DataFrame.

    Args:
        plip_obj (PDBComplex): The PLIP object with interactions.
        ligand_code (str): The ligand code.

    Returns:
        pd.DataFrame: The interactions between the protein and ligand.
    """
    all_interactions = []
    for lig_name, interaction_set in plip_obj.interaction_sets.items():
        if ligand_code in lig_name:
            for interaction in interaction_set.all_itypes:
                try:
                    all_interactions.append(parse_interaction(interaction))
                except NotImplementedError as e:
                    print(e)
                    continue
            break
    interactions_df = _interactions_to_dataframe(all_interactions)
    return interactions_df


def combine_complex(protein_file: str, ligand_file: str) -> str:
    """
    Combine the protein and ligand files into a single complex file.

    Args:
        protein_file (str): The path to the protein file.
        ligand_file (str): The path to the ligand file.

    Returns:
        str: The combined complex file as a string.
    """
    cleaned_protein_file = "\n".join(
        [line for line in open(protein_file).readlines() if line.startswith("ATOM")]
    )
    cmd.read_pdbstr(cleaned_protein_file, "protein_")
    cmd.load(ligand_file, "ligand_")
    cmd.create("complex", "protein_ or ligand_")
    pdbstr = cmd.get_pdbstr("complex")
    cmd.delete("all")
    return pdbstr


def prepare_plipobj(
    protein_file: str = None,
    ligand_file: str = None,
    complex_file: str = None,
    pymol_visualization: bool = False,
):
    """
    Creates PLIP object with interactions between protein and ligand specified
    in the input files.

    Args:
        protein_file (str): The path to the protein PDB file.
        ligand_file (str): The path to the ligand MOL or SDF file.
        complex_file (str): The path to the complex PDB file (if already
        exists).
        pymol_visualization (bool): Whether to visualize the interactions in
        PyMOL.

    Returns:
        PDBComplex: The PLIP object with calculated interactions.
    """
    if complex_file:
        plip_obj = PDBComplex()
        plip_obj.load_pdb(complex_file, as_string=False)
        plip_obj.analyze()
        return plip_obj
    else:
        pdbstr = combine_complex(protein_file, ligand_file)
        plip_obj = PDBComplex()
        plip_obj.load_pdb(pdbstr, as_string=True)
        plip_obj.analyze()
        return plip_obj


def strip_zero_at_end(string: str) -> str:
    """
    Strip the zero at the end of a string. Is recursive until the string does
    only ends with 1 decimal point and 0 or no 0.

    Args:
        string (str): The string to strip.

    Returns:
        str: The stripped string.
    """
    if string[-1] == "0" and string[-2] != ".":
        return strip_zero_at_end(string[:-1])
    else:
        return string


def get_backbone_protein_coords(protein_file: str) -> dict:
    """
    Get the backbone protein coordinates from the protein file.

    Args:
        protein_file (str): The path to the protein file.

    Returns:
        dict: The protein coordinates with 1 for backbone atoms and 0 for
        non-backbone atoms.
    """
    coords = {}
    # example = "[((60.971, 25.868, 120.62), 6)]"
    protein = open(protein_file, "r").read().split("\n")
    for line in protein:
        if line.startswith("ATOM"):
            key = (
                float(strip_zero_at_end(line[30:38].strip())),
                float(strip_zero_at_end(line[38:46].strip())),
                float(strip_zero_at_end(line[46:54].strip())),
            )
            if line[12:16].strip() in ["CA", "C", "N", "O"]:
                coords[key] = 0
            else:
                coords[key] = 1
    return coords


def filter_out_backbone_interactions(
    interactions_df: pd.DataFrame, protein_file: str
) -> pd.DataFrame:
    """
    Filter out backbone interactions from the interactions DataFrame.

    Args:
        interactions_df (pd.DataFrame): The interactions DataFrame.
        protein_file (str): The path to the protein file.

    Returns:
        pd.DataFrame: The interactions DataFrame with backbone interactions removed.
    """
    protein_coords = get_backbone_protein_coords(protein_file)
    protein_atoms = [i[0][0] for i in interactions_df["protein_atom"]]
    for i, atom in enumerate(protein_atoms):
        if atom in protein_coords:
            if protein_coords[atom] == 0:
                interactions_df.drop(i, inplace=True)
    return interactions_df


def filter_out_multiple_interaction_residues(
    interactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter out residues with many interactions from the interactions DataFrame.

    Args:
        interactions_df (pd.DataFrame): The interactions DataFrame.

    Returns:
        pd.DataFrame: The interactions DataFrame with residues with many interactions removed.
    """
    interactions_df["count"] = interactions_df.groupby(
        ["protein_number", "protein_chain", "protein_residue"]
    )["protein_chain"].transform("count")
    interactions_df = interactions_df[interactions_df["count"] < 2]
    # pandas does not like dropping on a copy of a slice of a DataFrame
    interactions_df = interactions_df.copy()
    interactions_df.drop(columns=["count"], inplace=True)

    return interactions_df
