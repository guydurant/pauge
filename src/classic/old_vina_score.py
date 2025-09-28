import warnings

warnings.warn = lambda *args, **kwargs: None

from meeko import MoleculePreparation, PDBQTWriterLegacy, RDKitMolCreate, PDBQTMolecule
import vina
import pypdb
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from posebusters.modules.rmsd import check_rmsd
import os
import tempfile
import numpy as np
from tqdm import tqdm

DEFAULT_CONFIG = {
    "box_size": [25, 25, 25],
    "exhaustiveness": 12,
    "n_poses": 20,
    "cpu": 20,
}


def read_file(file):
    if file.endswith(".pdb"):
        return Chem.MolFromPDBFile(file)
    elif file.endswith(".mol"):
        return Chem.MolFromMolFile(file)
    elif file.endswith(".sdf"):
        return Chem.MolFromMolFile(file)
    elif file.endswith(".mol2"):
        return Chem.MolFromMol2File(file)
    else:
        raise ValueError("Unsupported file format", file.split(".")[-1])


def get_centroid(mol):
    coords = []
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        coords.append(conf.GetAtomPosition(i))
    centroid = np.mean(coords, axis=0)
    return centroid


def prepare_ligand(mol):
    mol = Chem.AddHs(mol)
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    for setup in mol_setups:
        pdbqt_str, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
    return pdbqt_str


def prepare_receptor(pdbblock):
    pdbblock = "\n".join(
        [line for line in pdbblock.splitlines()]# if line.startswith("ATOM")]
    )
    ob_mol = pybel.readstring("pdb", pdbblock, opt={"flex": False})
    # print(pybel.outformats)
    ob_mol.addh()
    # create pdbqt string
    pdbqt_str = ob_mol.write("pdbqt")
    # keep only ATOM and HETATM records
    pdbqt_str = "\n".join(
        [line for line in pdbqt_str.splitlines() if line.startswith("ATOM") or line.startswith("HETATM")]
    )
    print(pdbqt_str)
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as temp_file:
        temp_file.write(pdbqt_str.encode("utf-8"))
        temp_file_path = temp_file.name
    return temp_file_path




def vina_score(receptor, ligand_code, ligand_file=None, config=DEFAULT_CONFIG):
    scorer = vina.Vina(sf_name="vina", verbosity=2)
    ligand = read_file(ligand_file)
    centroid = get_centroid(ligand)
    # if receptor is a file path, read it
    if not "\n" in receptor:
        receptor = open(receptor, "r").read()
    ligand_pdbqt = prepare_ligand(ligand)
    receptor_pdbqt = prepare_receptor(receptor)
    scorer.set_receptor(receptor_pdbqt)
    scorer.set_ligand_from_string(ligand_pdbqt)
    scorer.compute_vina_maps(
        box_size=config["box_size"],
        center=list(centroid),
    )
    score = scorer.score()
    os.system(f"rm {receptor_pdbqt}")
    return score[0]


if __name__ == "__main__":
    import click
    from denovopocketmetrics.src.utils.interface import display_single_value

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
        """Command line interface for the Vina Score metric.

        Args:
            generated_prot_file (str): The path to the generated protein file.
            ligand_code (str): The ligand code.
            ligand_file (str): The path to the ligand file.
        """
        if not ligand_code and not ligand_file:
            raise click.UsageError(
                "At least one of --ligand_code or --ligand_file must be provided."
            )

        value = vina_score(generated_prot_file, ligand_code, ligand_file=ligand_file)
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value,
            value_name="Vina Score (kcal/mol)",
        )

    main()
