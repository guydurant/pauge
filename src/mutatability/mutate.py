from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS
import pandas as pd
from molecular_rectifier import Rectifier
import click
from denovopocketmetrics.src.utils.interactions import prepare_plipobj, get_interactions
from denovopocketmetrics.src.mutatability.commands import commands
from tqdm import tqdm
import os
RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings

MUTATION = {
    # Nitrogen to Carbon
    6: 7,
    # Oxygen to Carbon
    8: 6,
    # Nitrogen to Carbon
    7: 6,
}


def adjust_valence(mol_editable, atom_idx):
    atom = mol_editable.GetAtomWithIdx(atom_idx)
    atom_valences = {
        1: 1,  # Hydrogen
        6: 4,  # Carbon
        7: 3,  # Nitrogen
        8: 2,  # Oxygen
        9: 1,  # Fluorine
        15: 3,  # Phosphorus
        16: 2,  # Sulfur
        17: 1,  # Chlorine
    }
    if atom.GetAtomicNum() in atom_valences:
        expected_valence = atom_valences[atom.GetAtomicNum()]
        current_valence = atom.GetTotalValence()
        if current_valence > expected_valence:
            for neighbor in atom.GetNeighbors():
                if (
                    mol_editable.GetBondBetweenAtoms(
                        atom.GetIdx(), neighbor.GetIdx()
                    ).GetBondType()
                    == Chem.BondType.DOUBLE
                ):
                    mol_editable.GetBondBetweenAtoms(
                        atom.GetIdx(), neighbor.GetIdx()
                    ).SetBondType(Chem.BondType.SINGLE)
                    break
    return mol_editable


def change_interaction_atom(
    df_row: pd.Series, lig_mol: Chem.Mol, mutation_type: str = "change"
) -> Chem.Mol:
    """
    Alter a ligand atom to an atom type unable to make a specific interaction

    Args:
        df_row (pd.Series): The row of the interaction
        lig_mol (Chem.Mol): The ligand molecule#
    Returns:
        Chem.Mol: The mutated ligand molecule
    """
    lig_mol_copy = Chem.Mol(lig_mol)
    ligand_id, ligand_atom = df_row["ligand_id"], df_row["ligand_atom"]
    for atom in lig_mol.GetAtoms():
        if ligand_id == atom.GetIdx() + 1:
            mutatable_atom = atom
            break
    lig_mol_copy = Chem.RemoveHs(lig_mol_copy)
    if mutation_type == "change":
        lig_mol_copy.GetAtomWithIdx(mutatable_atom.GetIdx()).SetAtomicNum(
            MUTATION[mutatable_atom.GetAtomicNum()]
        )
    elif mutation_type == "add":
        mol_editable = Chem.RWMol(lig_mol_copy)
        carbon_idx = mol_editable.AddAtom(Chem.Atom(6))
        mol_editable.AddBond(mutatable_atom.GetIdx(), carbon_idx, Chem.BondType.SINGLE)
        mol_editable.GetAtomWithIdx(carbon_idx).SetNumExplicitHs(3)
        mol_editable.UpdatePropertyCache(strict=False)
        mol_editable = adjust_valence(mol_editable, mutatable_atom.GetIdx())
        mol_editable = Chem.AddHs(mol_editable)
        Chem.SanitizeMol(mol_editable)
        Chem.Kekulize(mol_editable)
        mcs_result = rdFMCS.FindMCS(
            [mol_editable, lig_mol_copy],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=True,
        )
        core = Chem.MolFromSmarts(mcs_result.smartsString)
        substruct = Chem.AllChem.DeleteSubstructs(
            Chem.AllChem.ReplaceSidechains(Chem.RemoveHs(lig_mol_copy), core),
            Chem.MolFromSmiles("*"),
        )
        substruct.UpdatePropertyCache()
        Chem.AllChem.ConstrainedEmbed(
            mol_editable,
            substruct,
        )
        lig_mol_copy = mol_editable
    # elif mutation_type == "remove":
    #     mol_editable = Chem.RWMol(lig_mol_copy)
    #     mol_editable.RemoveAtom(mutatable_atom.GetIdx())
    #     mol_editable.UpdatePropertyCache(strict=False)
    #     mol_editable = mol_editable.GetMol()
    #     mol_editable = Chem.AddHs(mol_editable)
    #     # Chem.SanitizeMol(mol_editable)
    #     print(Chem.MolToSmiles(mol_editable))
    #     lig_mol_copy = mol_editable
    # recto = Rectifier(lig_mol_copy, valence_correction="charge")
    # recto.fix()
    # lig_mol_copy = recto.mol
    Chem.SanitizeMol(lig_mol_copy)
    lig_mol_copy = Chem.RemoveHs(lig_mol_copy)
    if lig_mol_copy is None:
        raise ValueError(f"Could not mutate atom {ligand_atom} in ligand {ligand_id}")
    return lig_mol_copy


def mutate_all(
    protein_file,
    ligand_file,
    ground_truth_ligand,
    mutation_type="change",
    complex_file=None,
    pymol_visualization=False,
):
    """
    Alter all ligand atoms in a set of interactions to an atom type unable to make that interaction

    Args:
        lig_mol (Chem.Mol): The ligand molecule
        interactions (pd.DataFrame): The interactions to be mutated

    Returns:
        dict: A dictionary of mutated ligands, with the key being the unique interaction identifier
    """
    plipobj = prepare_plipobj(
        protein_file, ligand_file, complex_file, pymol_visualization
    )
    # get last atom of protein
    interactions = get_interactions(plipobj)
    lig_mol = Chem.MolFromMolFile(ground_truth_ligand)
    if lig_mol is None:
        raise ValueError(f"Could not read ground truth ligand {ground_truth_ligand}")
    mutated_ligands = {}
    for i, row in interactions.iterrows():
        if row["type"] in ["hydrophobic", "hbond"]:
            # only mutate hydrophobic and hydrogen bond interactions for now
            # TODO add more interaction types
            try:
                key = f"{row['type']}_{row['protein_chain']}_{row['protein_residue']}_{row['protein_number']}_{row['ligand_id']}"
                mutated_ligand = change_interaction_atom(row, lig_mol, mutation_type)
                mutated_ligands[key] = mutated_ligand
            except Exception as e:
                print("Could not mutate", key, e)
                continue
    return mutated_ligands


def design_for_mutant(
    model,
    mutation_type,
    generated_protein_file,
    generated_ligand_file,
    ground_truth_protein_file,
    ground_truth_ligand_file,
    out_folder,
):
    command = commands[model]
    mutant_mols = mutate_all(
        generated_protein_file, generated_ligand_file, ground_truth_ligand_file, mutation_type
    )
    for key, ligand in mutant_mols.items():
        mutant_ligand_file = f"{out_folder}/{key}.sdf"
        Chem.MolToMolFile(ligand, mutant_ligand_file)
        if not os.path.exists(f"{out_folder}/{key}"):
            os.makedirs(f"{out_folder}/{key}")
        if os.path.exists(f"{out_folder}/{key}" + "/packed/input_packed_1_1.pdb") or os.path.exists(f"{out_folder}/{key}"+ "/0_whole.pdb"):
            # print(f"Aleady designed {key}")
            continue
        complete_command = command(
            ground_truth_protein_file, mutant_ligand_file, f"{out_folder}/{key}"
        )
        os.system(complete_command)


@click.command()
@click.option("--model", type=str, help="The model to use for design", required=True)
@click.option("--mutation_type", type=str, help="The mutation type", required=True, default="change")
@click.option(
    "--generated_protein_file",
    type=str,
    help="The generated protein file",
    required=False,
)
@click.option(
    "--generated_ligand_file",
    type=str,
    help="The generated ligand file",
    required=False,
)
@click.option(
    "--ground_truth_protein_file",
    type=str,
    help="The ground truth protein file",
    required=False,
)
@click.option(
    "--ground_truth_ligand_file",
    type=str,
    help="The ground truth ligand file",
    required=False,
)
@click.option("--out_folder", type=str, help="The output folder", required=False)
@click.option(
    "--csv_file",
    type=str,
    help="The csv file containing the generated protein and ligand files",
    required=False,
)
def main(
    model,
    mutation_type,
    generated_protein_file,
    generated_ligand_file,
    ground_truth_protein_file,
    ground_truth_ligand_file,
    out_folder,
    csv_file,
):
    if csv_file:
        df = pd.read_csv(csv_file)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            out_folder_specific = f"{out_folder}/{row['code']}"
            if not os.path.exists(out_folder_specific):
                os.makedirs(out_folder_specific)
            try:
                design_for_mutant(
                    model,
                    mutation_type,
                    row["protein"],
                    row["ligand"],
                    row["ground_truth_protein"],
                    row["ground_truth_ligand"],
                    out_folder_specific,
                )
            except Exception as e:
                print(e)
    else:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        design_for_mutant(
            model,
            generated_protein_file,
            generated_ligand_file,
            ground_truth_protein_file,
            ground_truth_ligand_file,
            out_folder,
        )


if __name__ == "__main__":
    main()
