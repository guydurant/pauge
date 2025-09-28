from rdkit import Chem
from pymol import cmd
import pypdb


def read_rdkit(file: str) -> Chem.Mol:
    """Read a molecule from a file.

    Args:
        file (str): The path to the file.

    Returns:
        Chem.Mol: The molecule.
    """
    if file.endswith(".sdf"):
        return Chem.SDMolSupplier(file)[0]
    elif file.endswith(".mol"):
        return Chem.MolFromMolFile(file)
    elif file.endswith(".pdb"):
        return Chem.MolFromPDBFile(file)
    elif file.endswith(".mol2"):
        return Chem.MolFromMol2File(file)
    else:
        raise ValueError("Unsupported file format for RDKit.")


def get_ligand_smiles(ligand_code: str) -> str:
    """Get the SMILES string for the ligand code from the PDB.

    Args:
        ligand_code (str): The ligand code.

    Returns:
        str: The SMILES string.
    """
    chem_desc = pypdb.describe_chemical(ligand_code)
    return [
        i["descriptor"]
        for i in chem_desc["pdbx_chem_comp_descriptor"]
        if i["type"] == "SMILES"
    ][0]


def get_molblocks(
    prot_file: str,
    ligand_code: str,
    cutoff: int = 5,
    ligand_file: str = None,
):
    """Get the PDB blocks for the protein and ligand.

    Args:
        prot_file (str): The path to the protein PDB file.
        ligand_code (str): The ligand code.
        cutoff (int): The cutoff distance for the pocket selection.
        ligand_file (str): The path to the ligand MOL or SDF file.

    Returns:
        tuple: The PDB blocks for the protein and ligand.
    """
    prot_block, lig_block = separate_pdb(prot_file, ligand_code)
    if ligand_file:
        lig_block = Chem.MolToPDBBlock(Chem.MolFromMolFile(ligand_file, sanitize=False))
    cmd.reinitialize()
    cmd.read_pdbstr(prot_block, "prot")
    cmd.read_pdbstr(lig_block, "lig")
    if cutoff == -1:
        cmd.select("pocket", "prot")
    else:
        cmd.select(
            "pocket",
            "prot and byres lig around " + str(cutoff),
        )
    lig_gen_block = cmd.get_pdbstr("lig")
    prot_gen_block = cmd.get_pdbstr("pocket")
    cmd.delete("all")
    return prot_gen_block, lig_gen_block


def separate_pdb(pdb_file, ligand_code="UNK"):
    """
    Separate the protein and ligand blocks from a PDB file.

    Args:
        pdb_file (str): The path to the PDB file.
        ligand_code (str): The ligand code.

    Returns:
        tuple: The protein and ligand blocks.
    """
    with open(
        pdb_file,
        "r",
    ) as f:
        lines = f.read().splitlines()
    prot_block = [line for line in lines if line.startswith("ATOM")]
    lig_block = []
    found_ligand = False
    current_chain = None

    for line in lines:
        if line.startswith("HETATM") and line[17:20].strip() == ligand_code:
            chain_id = line[21]  # The chain identifier is usually at position 22
            if not found_ligand:
                lig_block.append(line)
                current_chain = chain_id
            elif chain_id != current_chain:
                found_ligand = True
                # Stop collecting once the same ligand code on a different chain is encountered
                break
        elif (
            lig_block
            and line.startswith("HETATM")
            and line[17:20].strip() != ligand_code
        ):
            # Stop collecting once a different ligand is encountered
            break
        elif lig_block and not line.startswith("HETATM"):
            # Stop collecting once any other type of record is encountered
            break
    lig_block_str = "\n".join(lig_block)
    prot_block_str = "\n".join(prot_block)
    return prot_block_str, lig_block_str
