from Bio import PDB
import io
import numpy as np
from src.utils.structure import get_molblocks


class StringIOOutput:
    """A class to capture the output of a StringIO object.
    Allows parsing into Bio.PDB without writing to disk."""

    def __init__(self):
        self.content = ""

    def write(self, text):
        self.content += text


ATOM_RADII = {
    #    "H": 1.20,  # Who cares about hydrogen??
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "F": 1.47,
    "P": 1.80,
    "CL": 1.75,
    "MG": 1.73,
}


def _count_clashes(pdbblock: str, clash_cutoff: float = 0.63):
    """Count the number of clashes in a PDB block.
    Code written by Dr. Brennan Abandes
    - https://www.blopig.com/blog/2023/05/checking-your-pdb-file-for-clashing-atoms/.

    Args:
        pdbblock (str): The PDB block.
        clash_cutoff (float): The VDW radius ratio cutoff for clashes.

    Returns:
        int: The number of clashes.
    """
    parser = PDB.PDBParser(QUIET=True)
    io_file = io.StringIO(pdbblock)
    structure = parser.get_structure("example", io_file)
    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {
        i + "_" + j: (clash_cutoff * (ATOM_RADII[i] + ATOM_RADII[j]))
        for i in ATOM_RADII
        for j in ATOM_RADII
    }
    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in ATOM_RADII]
    coords = np.array([a.coord for a in atoms], dtype="d")
    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)
    # Initialize a list to hold clashes
    clashes = []
    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(
            np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values())
        )
        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]
        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue
            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (
                atom_2.name == "N" and atom_1.name == "C"
            ):
                continue
            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue
            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))
    # print residues where clashes occur
    for clash in clashes:
        res_1 = clash[0].parent
        res_2 = clash[1].parent
        # print(
        #     f"Clash between {clash[0].name} {res_1.get_resname()} {res_1.get_id()} {res_1.get_parent().id} and {clash[1].name} {res_2.get_resname()} {res_2.get_id()} {res_1.get_parent().id}"
        # )
    return len(clashes) // 2


def count_clashes(
    generated_prot_file: str,
    ligand_code: str,
    cutoff: int = 5,
    ligand_file: str = None,
):
    """Wrapper function that count the number of clashes in a generated protein pocket.

    Args:
        generated_prot_file (str): The path to the generated protein file.
        ligand_code (str): The ligand code.
        cutoff (int): The cutoff distance for the pocket selection.
        ligand_file (str): The path to the ligand MOL or SDF file.

    Returns:
        int: The number of clashes.
    """
    prot_gen_block, lig_gen_block = get_molblocks(
        generated_prot_file,
        ligand_code,
        6,  # higher cutoff to include more atoms on the pocket
        ligand_file,
    )

    prot_gen_block_clean = (
        "\n".join(
            [line for line in prot_gen_block.split("\n") if line.startswith("ATOM")]
        )
        + "\n"
    )
    return _count_clashes(prot_gen_block_clean)


if __name__ == "__main__":
    import click
    from src.utils.interface import display_single_value

    @click.command()
    @click.option(
        "--generated_prot_file",
        "-g",
        required=True,
        type=str,
        help="Path to the generated protein file.",
    )
    @click.option(
        "--ligand_code",
        "-c",
        required=False,
        type=str,
        help="The ligand code.",
    )
    @click.option(
        "--ligand_file",
        "-l",
        required=False,
        type=str,
        help="Path to the ligand file.",
    )
    def main(generated_prot_file, ligand_code, ligand_file):
        """Command line interface for the Side Chain No of Clashes metric.

        Args:
            generated_prot_file (str): The path to the generated protein file.
            ligand_code (str): The ligand code.
            ligand_file (str): The path to the ligand file.
        """
        if not ligand_code and not ligand_file:
            raise click.UsageError(
                "At least one of --ligand_code or --ligand_file must be provided."
            )

        value = count_clashes(generated_prot_file, ligand_code, ligand_file=ligand_file)
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value,
            value_name="Number of Side Chain Clashes",
        )

    main()
