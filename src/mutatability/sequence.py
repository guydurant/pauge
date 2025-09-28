import pandas as pd
import click
from src.utils.interactions import (
    prepare_plipobj,
    get_interactions,
    filter_out_backbone_interactions,
    filter_out_multiple_interaction_residues,
)
from tqdm import tqdm
import numpy as np
from scipy import stats
import os

INPUT_FILES = {
    "ligandmpnn": "packed/input_packed_1_1.pdb",
    "pocketgen": "0_whole.pdb",
}


def check_mutation(key, pdb_file):
    pdblines = open(pdb_file, "r").readlines()
    for line in pdblines:
        if line.startswith("ATOM"):
            if (
                line[21] == key.split("_")[1]
                and line[22:26].strip() == key.split("_")[3]
            ):
                if line[17:20] == key.split("_")[2]:
                    return 0
                else:
                    return 1
    return 0


def sequence_all(
    protein_file,
    ligand_file,
    ground_truth_ligand,
    input_folder,
    generic_file,
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
    interactions = get_interactions(plipobj)
    filtered_interactions = filter_out_backbone_interactions(interactions, protein_file)
    filtered_interactions = filter_out_multiple_interaction_residues(
        filtered_interactions
    )
    mutations = []
    background_mutations = []
    for i, row in filtered_interactions.iterrows():
        # if row["type"] in ["hydrophobic", "hbond"]:
        if row["type"] in ["hbond"]:
            # only mutate hydrophobic and hydrogen bond interactions for now
            # TODO add more interaction types
            try:
                key = f"{row['type']}_{row['protein_chain']}_{row['protein_residue']}_{row['protein_number']}_{row['ligand_id']}"
                # get another random row that is not the same as the current row
                if len(filtered_interactions) == 1:
                    random_row = None
                else:
                    for _ in range(100):
                        random_row = filtered_interactions.sample()
                        random_key = f"{random_row['type'].values[0]}_{random_row['protein_chain'].values[0]}_{random_row['protein_residue'].values[0]}_{random_row['protein_number'].values[0]}_{random_row['ligand_id'].values[0]}"
                        if random_key != key:
                            if os.path.exists(
                                f"{input_folder}/{random_key}/{generic_file}"
                            ):
                                break
                    else:
                        random_row = None

                pdbfile = f"{input_folder}/{key}/{generic_file}"
                value = check_mutation(key, pdbfile)
                # print(f"Mutation: {key} - {value}")
                mutations.append(value)
                if random_row is None:
                    continue
                random_pdbfile = f"{input_folder}/{random_key}/{generic_file}"
                background_value = check_mutation(key, random_pdbfile)
                background_mutations.append(background_value)
            except Exception as e:
                # print(f"Error: {e}")
                pass
    return mutations, background_mutations


@click.command()
@click.option(
    "--model",
    type=click.Choice(["ligandmpnn", "pocketgen"]),
    help="The model used to generate the protein",
    required=True,
)
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
    generated_protein_file,
    generated_ligand_file,
    ground_truth_protein_file,
    ground_truth_ligand_file,
    out_folder,
    csv_file,
):
    if csv_file:
        all_mutations = []
        all_mutations_background = []
        df = pd.read_csv(csv_file)
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Checking mutations"):
            try:
                out_folder_specific = f"{out_folder}/{row['code']}"
                mutations, background_mutations = sequence_all(
                    row["protein"],
                    row["ligand"],
                    row["ground_truth_ligand"],
                    out_folder_specific,
                    INPUT_FILES[model],
                )
                if len(mutations) > 0:
                    all_mutations.append(np.mean(mutations))
                if len(background_mutations) > 0:
                    all_mutations_background.append(np.mean(background_mutations))
            except Exception as e:
                print(f'File {row["code"]} failed with error: {e}')
        confidence_mut = stats.t.interval(
            0.95,
            len(all_mutations) - 1,
            loc=np.mean(all_mutations),
            scale=stats.sem(all_mutations),
        )
        confidence_back = stats.t.interval(
            0.95,
            len(all_mutations_background) - 1,
            loc=np.mean(all_mutations_background),
            scale=stats.sem(all_mutations_background),
        )
        print(
            f"Mean Mutations: {np.mean(all_mutations)}+-{np.mean(all_mutations)-confidence_mut[0]}",
        )
        print(
            f"Mean Background Mutations: {np.mean(all_mutations_background)}+-{np.mean(all_mutations_background)-confidence_back[0]}",
        )
    else:
        mutations, background_mutations = sequence_all(
            generated_protein_file,
            generated_ligand_file,
            ground_truth_ligand_file,
            out_folder,
            INPUT_FILES[model],
        )
        print(f"Mutations: {mutations}")
        print(f"Background Mutations: {background_mutations}")


if __name__ == "__main__":
    main()
