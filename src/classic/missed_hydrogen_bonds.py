from denovopocketmetrics.src.utils.interactions import prepare_plipobj, get_interactions
from denovopocketmetrics.src.utils.structure import read_rdkit
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA
import click
from denovopocketmetrics.src.utils.interface import display_single_value


def get_hbonds(
    protein_file,
    ligand_file,
    complex_file=None,
    pymol_visualization=False,
):
    """
    Get all hydrogen bonds in a set of interactions

    Args:
        protein_file (str): The protein file
        ligand_file (str): The ligand file

        complex_file (str): The complex file
        pymol_visualization (bool): Whether to visualize the interactions in PyMOL

    Returns:
        pd.DataFrame: A DataFrame of hydrogen bond interactions
    """
    plipobj = prepare_plipobj(
        protein_file, ligand_file, complex_file, pymol_visualization
    )
    # get last atom of protein
    interactions = get_interactions(plipobj)
    return interactions[interactions["type"] == "hbond"]


def get_hbond_acceptors_and_donors(ligand_file: str):
    """
    Get the acceptor and donor atoms from a ligand molecule

    Args:
        ligand_file (str): The ligand file

    Returns:
        tuple: The acceptor and donor atoms
    """
    mol = read_rdkit(ligand_file)
    return CalcNumHBD(mol), CalcNumHBA(mol)


def missed_hydrogen_bonds(
    protein_file,
    ligand_file,
    complex_file=None,
    pymol_visualization=False,
):
    """
    Get the missed hydrogen bonds in a set of interactions

    Args:
        protein_file (str): The protein file
        ligand_file (str): The ligand file
        ground_truth_ligand (str): The ground truth ligand file
        complex_file (str): The complex file
        pymol_visualization (bool): Whether to visualize the interactions in PyMOL

    Returns:
        pd.DataFrame: A DataFrame of missed hydrogen bond interactions
    """
    hbonds = get_hbonds(
        protein_file,
        ligand_file,
        complex_file,
        pymol_visualization,
    )
    num_hbonds = len(hbonds)
    num_acceptors, num_donors = get_hbond_acceptors_and_donors(ligand_file)
    return num_donors + num_acceptors - num_hbonds


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

    value = missed_hydrogen_bonds(generated_prot_file, ligand_file)
    display_single_value(
        generated_prot_file,
        ligand_code,
        ligand_file,
        value,
        value_name="Missing Hydrogen Bonds",
    )


if __name__ == "__main__":
    main()
