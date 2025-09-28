from src.utils.structure import get_molblocks, get_ligand_smiles
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from pathlib import Path
from posebusters import PoseBusters
import os
from typing import List, Tuple

RDLogger.DisableLog("rdApp.*")


def bust_complex(
    generated_prot_file: str, ligand_code: str, ligand_file: str = None
) -> Tuple[List[str], List[str]]:
    """Run PoseBusters on a protein-ligand complex. Separates the ligand and protein-ligand failures.

    Args:
        generated_prot_file (str): The path to the generated protein file.
        ligand_code (str): The ligand code.
        ligand_file (str): The path to the ligand file.

    Returns:
        Tuple[List[str], List[str]]: The ligand and protein-ligand failures.
    """
    prot_gen_block, lig_gen_block = get_molblocks(
        generated_prot_file,
        ligand_code,
        ligand_file=ligand_file,
    )
    if ligand_file:
        lig_gen = Chem.MolFromMolFile(ligand_file, sanitize=False)
    else:
        lig_template_smiles = get_ligand_smiles(ligand_code)
        lig_temp = Chem.MolFromSmiles(lig_template_smiles)

        lig_gen = Chem.MolFromPDBBlock(
            "\n".join(
                [
                    line
                    for line in lig_gen_block.split("\n")
                    if line.startswith("HETATM")
                ]
            )
            + "\n"
            + "END"
            + "\n",
            sanitize=False,
        )
        if lig_gen is None:
            raise Exception("lig_gen is None")
        lig_gen = AllChem.AssignBondOrdersFromTemplate(lig_temp, lig_gen)
    buster = PoseBusters(config="dock")
    df = buster.bust(
        [lig_gen], lig_gen, Chem.MolFromPDBBlock(prot_gen_block, sanitize=False)
    )
    lig_fail = []
    complex_fail = []
    columns = ["minimum_distance_to_protein", "volume_overlap_with_protein"]
    for col in df.columns:
        if False in df[col].values:
            if col in columns:
                complex_fail.append(col)
            else:
                lig_fail.append(col)
    return lig_fail, complex_fail


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
        """Command line interface for the PoseBusters plausability metric.

        Args:
            generated_prot_file (str): The path to the generated protein file.
            ligand_code (str): The ligand code.
            ligand_file (str): The path to the ligand file.
        """

        if not ligand_code and not ligand_file:
            raise click.UsageError(
                "At least one of --ligand_code or --ligand_file must be provided."
            )

        lig_fail, complex_fail = bust_complex(
            generated_prot_file, ligand_code, ligand_file=ligand_file
        )
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value=[len(lig_fail) == 0, len(complex_fail) == 0],
            value_name=["Ligand PBValid", "Protein-Ligand PBValid"],
        )

    main()
