from denovopocketmetrics.src.classic.vina_score import vina_score
from denovopocketmetrics.src.classic.amino_acid_recovery import aarecovery
from denovopocketmetrics.src.classic.missed_hydrogen_bonds import missed_hydrogen_bonds
from denovopocketmetrics.src.classic.interaction_similarity import (
    interaction_similarity,
)
from denovopocketmetrics.src.utils.structure import get_molblocks
import pandas as pd
from tqdm import tqdm
import numpy as np
import click
from joblib import Parallel, delayed
from denovopocketmetrics.src.utils.interface import (
    display_single_value,
    display_dataset_values,
)
from warnings import filterwarnings

filterwarnings("ignore")


def process_csv(csv_file: str):
    """Process the CSV file for the plausability metrics.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        Tuple: A tuple of lists containing the codes, protein files, ligand codes, ligand files, ground truth protein files, and ground truth ligand files.
    """
    df = pd.read_csv(csv_file)
    if "protein" not in df.columns:
        raise ValueError("The CSV file must contain a column named 'protein'.")
    if "ligand" not in df.columns and "ligand_code" not in df.columns:
        raise ValueError(
            "The CSV file must contain a column named 'ligand' or 'ligand_code'."
        )
    codes = df["code"].tolist()
    protein_files = df["protein"].tolist()
    # if "ligand_code" in df.columns:
    #     ligand_codes = df["ligand_code"].tolist()
    # else:
    ligand_codes = [None for _ in protein_files]
    ligand_files = (
        df["ligand"].tolist() if "ligand" in df else [None for _ in protein_files]
    )
    ground_truth_prot_files = df["ground_truth_protein"].tolist()
    ground_truth_lig_files = df["ground_truth_ligand"].tolist()
    return (
        codes,
        protein_files,
        ligand_codes,
        ligand_files,
        ground_truth_prot_files,
        ground_truth_lig_files,
    )


@click.command()
@click.option(
    "--csv_file",
    "-c",
    required=False,
    type=str,
    help="Path to the CSV file.",
)
@click.option(
    "--generated_prot_file",
    "-g",
    required=False,
    type=str,
    help="Path to the generated protein file.",
)
@click.option(
    "--ligand_code",
    "-x",
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
@click.option(
    "--ground_truth_prot_file",
    "-t",
    required=False,
    type=str,
    help="Path to the ground truth protein file.",
)
@click.option(
    "--ground_truth_lig_file",
    "-p",
    required=False,
    type=str,
    help="Path to the ground truth ligand file.",
)
@click.option(
    "--output",
    "-o",
    required=False,
    type=str,
    help="Path to the output CSV file.",
)
def main(
    csv_file,
    generated_prot_file,
    ligand_code,
    ligand_file,
    ground_truth_prot_file,
    ground_truth_lig_file,
    output,
):
    """Command line interface for all the classic metrics."""
    if not csv_file and not generated_prot_file:
        raise click.UsageError(
            "At least one of --csv_file or --generated_prot_file (and --ligand_code or --ligand_file) must be provided."
        )
    if generated_prot_file:
        if not ligand_code and not ligand_file:
            raise click.UsageError(
                "At least one of --ligand_code or --ligand_file must be provided."
            )
    if not csv_file:
        aa_recovery_value = aarecovery(
            generated_prot_file,
            ground_truth_prot_file,
            ground_truth_lig_file,
            ligand_code,
            ligand_file=ligand_file,
        )
        vina_score_value = vina_score(
            generated_prot_file, ligand_code, ligand_file=ligand_file
        )
        missed_hydrogen_bonds_value = missed_hydrogen_bonds(
            generated_prot_file, ligand_file
        )
        interaction_similarity_value = interaction_similarity(
            generated_prot_file,
            ground_truth_prot_file,
            ground_truth_lig_file,
            ligand_code,
            ligand_file=ligand_file,
        )
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value=[
                aa_recovery_value,
                vina_score_value,
                missed_hydrogen_bonds_value,
                interaction_similarity_value,
            ],
            value_name=[
                "Amino Acid Recovery",
                "Vina Score",
                "Missing Hydrogen Bonds",
                "Interaction Similarity",
            ],
        )
    else:
        (
            codes,
            protein_files,
            ligand_codes,
            ligand_files,
            ground_truth_prot_files,
            ground_truth_lig_files,
        ) = process_csv(csv_file)

        def process_protein(
            i,
            protein_file,
            ground_truth_prot_file,
            ground_truth_lig_file,
            ligand_code,
            ligand_file,
        ):
            result = {}
            try:
                result["aa_recovery"] = aarecovery(
                    protein_file,
                    ground_truth_prot_file,
                    ground_truth_lig_file,
                    ligand_code,
                    cutoff=5,
                    ligand_file=ligand_file,
                )
                result["vina_score"] = vina_score(
                    protein_file,
                    ligand_code,
                    ligand_file=ligand_file,
                )
                # result["missed_hydrogen_bonds"] = missed_hydrogen_bonds(
                #     protein_file,
                #     ligand_file,
                # )
                result["interaction_similarity"] = interaction_similarity(
                    protein_file,
                    ground_truth_prot_file,
                    ground_truth_lig_file,
                    ligand_code,
                    ligand_file=ligand_file,
                )
            except Exception as e:
                print(f"Error processing row {codes[i]}: {e}")
                return i, None  # Return the index and None if it failed
            return i, result  # Return the index and result if successful

        # Run the processing in parallel using joblib
        results = Parallel(n_jobs=-1)(
            delayed(process_protein)(
                i,
                protein_files[i],
                ground_truth_prot_files[i],
                ground_truth_lig_files[i],
                ligand_codes[i],
                ligand_files[i],
            )
            for i in tqdm(range(len(protein_files)), desc="Calculating classic metrics")
        )

        # Split successful results and failed indices
        values = {i: result for i, result in results if result is not None}
        failed = [i for i, result in results if result is None]
        # make 3 sf
        aa_recovery_value_mean = np.mean(
            [values[i]["aa_recovery"] for i in values]
        ).round(3)
        aa_recovery_value_std = np.std(
            [values[i]["aa_recovery"] for i in values]
        ).round(3)
        vina_score_value_mean = np.mean(
            [values[i]["vina_score"] for i in values]
        ).round(3)
        vina_score_value_std = np.std([values[i]["vina_score"] for i in values]).round(
            3
        )
        # missed_hydrogen_bonds_value_mean = np.mean(
        #     [values[i]["missed_hydrogen_bonds"] for i in values]
        # ).round(3)
        # missed_hydrogen_bonds_value_std = np.std(
        #     [values[i]["missed_hydrogen_bonds"] for i in values]
        # ).round(3)
        interaction_similarity_value_mean = np.mean(
            [values[i]["interaction_similarity"] for i in values]
        ).round(3)
        interaction_similarity_value_std = np.std(
            [values[i]["interaction_similarity"] for i in values]
        ).round(3)
        display_dataset_values(
            csv_file,
            values=[
                f"{aa_recovery_value_mean} ± {aa_recovery_value_std}",
                f"{vina_score_value_mean} ± {vina_score_value_std}",
                # f"{missed_hydrogen_bonds_value_mean} ± {missed_hydrogen_bonds_value_std}",
                f"{interaction_similarity_value_mean} ± {interaction_similarity_value_std}",
            ],
            value_names=[
                "Amino Acid Recovery < 5A (mean ± std)",
                "Vina Score (mean ± std)",
                # "Missing Hydrogen Bonds (mean ± std)",
                "Interaction Similarity (mean ± std)",
            ],
        )
        if output:
            output_df = pd.DataFrame(
                {
                    "code": [
                        codes[i] for i in range(len(protein_files)) if i not in failed
                    ],
                    "protein": [
                        protein_files[i]
                        for i in range(len(protein_files))
                        if i not in failed
                    ],
                    "ligand_code": [
                        ligand_codes[i]
                        for i in range(len(protein_files))
                        if i not in failed
                    ],
                    "ligand": [
                        ligand_files[i]
                        for i in range(len(protein_files))
                        if i not in failed
                    ],
                    "aa_recovery": [
                        values[i]["aa_recovery"] for i in values if i not in failed
                    ],
                    "vina_score": [
                        values[i]["vina_score"] for i in values if i not in failed
                    ],
                    # "missed_hydrogen_bonds": [
                    #     values[i]["missed_hydrogen_bonds"]
                    #     for i in values
                    #     if i not in failed
                    # ],
                    "interaction_similarity": [
                        values[i]["interaction_similarity"]
                        for i in values
                        if i not in failed
                    ],
                }
            )
            output_df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
