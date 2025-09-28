from src.utils.structure import get_molblocks
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem


AMINO_ACID_SMILES = {
    "ALA": "C",
    "ARG": "CCCNC(N)=N",
    "ASN": "CC(N)=O",
    "ASP": "CC(O)=O",
    "CYS": "CS",
    "GLN": "CCC(N)=O",
    "GLU": "CCC(O)=O",
    "GLY": "",  # Glycine has no side chain beyond the hydrogen, which is part of the backbone
    "HIS": "CC1=CNC=N1",
    "ILE": "C(C)CC",
    "LEU": "C(C)CC",
    "LYS": "CCCCN",
    "MET": "CSC",
    "PHE": "CC1=CC=CC=C1",
    "PRO": "C1CCN1",  # Proline is a bit unique, so its ring structure is considered its side chain
    "SER": "CO",
    "THR": "C(O)C",
    "TRP": "CC1=CNC2=C1C=CC=C2",
    "TYR": "CC1=CC=C(O)C=C1",
    "VAL": "C(C)C",
}


def protblock_to_aminoacids(protblock: str) -> dict:
    """Convert a protein block to a dictionary of amino acids.

    Args:
        protblock (str): The protein block.

    Returns:
        dict: A dictionary of amino acids.
    """
    amino_acids = {}
    for line in protblock.split("\n"):
        if line.startswith("ATOM"):
            # drop hydrogens
            if line.split()[-1] != "H":
                if line[13:16].strip() not in ["CA", "N", "C", "O", "OXT"]:
                    chain, resi, resn, alt = (
                        line[21],
                        line[22:26].strip(),
                        line[17:20].strip(),
                        line[26],
                    )
                    if f"{chain}_{resi}_{resn}_{alt}" not in amino_acids:
                        amino_acids[f"{chain}_{resi}_{resn}_{alt}"] = [line]
                    else:
                        amino_acids[f"{chain}_{resi}_{resn}_{alt}"].append(line)
    for aa in amino_acids:
        amino_acids[aa] = "\n".join(amino_acids[aa])
    return amino_acids


def bust_side_chains(
    generated_prot_file: str, ligand_code: str, ligand_file: str = None
) -> float:
    """Uses PoseBusters to check the validity of side chains of a generated pocket.

    Args:
        generated_prot_file (str): The path to the generated protein file.
        ligand_code (str): The ligand code.
        ligand_file (str): The path to the ligand file.

    Returns:
        float: The side chain PBValidity pass rate.
    """
    prot_gen_block, lig_gen_block = get_molblocks(
        generated_prot_file,
        ligand_code,
        ligand_file=ligand_file,
    )
    gen_amino_acids = protblock_to_aminoacids(prot_gen_block)
    gen_posebusters = 0
    for aa in gen_amino_acids:
        try:
            if aa.split("_")[2] in ["ALA", "GLY"]:
                continue
            if aa.split("_")[2] not in AMINO_ACID_SMILES:
                continue
            buster = PoseBusters(config="mol")
            temp = Chem.MolFromSmiles(AMINO_ACID_SMILES[aa.split("_")[2]])
            mol = Chem.MolFromPDBBlock(gen_amino_acids[aa])
            try:
                mol_clean = AllChem.AssignBondOrdersFromTemplate(temp, mol)
            except Exception:
                mol_clean = mol
            df = buster.bust([mol_clean], None, None)
            fail = False
            for col in df.columns:
                if False in df[col].values:
                    fail = True
            if fail:
                gen_posebusters += 1
        except Exception as e:
            gen_posebusters += 1
    return 1 - gen_posebusters / len(gen_amino_acids)


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
        """Command line interface for the side chain PBValidity metric.

        Args:
            generated_prot_file (str): The path to the generated protein file.
            ligand_code (str): The ligand code.
            ligand_file (str): The path to the ligand file.
        """
        if not ligand_code and not ligand_file:
            raise click.UsageError(
                "At least one of --ligand_code or --ligand_file must be provided."
            )

        value = bust_side_chains(
            generated_prot_file, ligand_code, ligand_file=ligand_file
        )
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value,
            value_name="Side Chain PBValidity Pass Rate ",
        )

    main()
