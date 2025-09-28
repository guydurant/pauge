from src.plausability.pbvalidity import bust_complex
from src.plausability.side_chain_clashes import count_clashes
from src.plausability.side_chain_pbvalidity import bust_side_chains
from src.utils.structure import get_molblocks
import pandas as pd
from tqdm import tqdm
import numpy as np
import click
from joblib import Parallel, delayed
from src.utils.interface import (
    display_single_value,
    display_dataset_values,
)


def process_csv(csv_file: str):
    """Process the CSV file for the plausability metrics.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        Tuple[List[str], List[str], List[str]]: The protein files, ligand codes, and ligand files.
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
    if "ligand_code" in df.columns:
        ligand_codes = df["ligand_code"].tolist()
    else:
        ligand_codes = [None for _ in protein_files]
    ligand_files = (
        df["ligand"].tolist() if "ligand" in df else [None for _ in protein_files]
    )
    gt_ligand_files = (
        df["ground_truth_ligand"] if "ground_truth_ligand" in df else [None] * len(codes)
    )
    return codes, protein_files, ligand_codes, gt_ligand_files, ligand_files, codes


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
    "--output",
    "-o",
    required=False,
    type=str,
    help="Path to the output CSV file.",
)
def main(csv_file, generated_prot_file, ligand_code, ligand_file, output):
    """Command line interface for all the plausability metrics."""
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
        lig_fail, complex_fail = bust_complex(
            generated_prot_file, ligand_code, ligand_file=ligand_file
        )
        side_chain_pbvalidity = bust_side_chains(
            generated_prot_file, ligand_code, ligand_file=ligand_file
        )
        side_chain_clashes = count_clashes(
            generated_prot_file, ligand_code, ligand_file=ligand_file
        )
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value=[
                len(lig_fail) == 0,
                len(complex_fail) == 0,
                side_chain_pbvalidity,
                side_chain_clashes,
            ],
            value_name=[
                "Ligand PBValid",
                "Protein-Ligand PBValid",
                "Side Chain PBValidity Pass Rate",
                "Side Chain No of Clashes",
            ],
        )
    else:
        codes, protein_files, ligand_codes, gt_ligand_files, ligand_files, codes = process_csv(csv_file)

        def process_protein(
            i,
            protein_file,
            ligand_code,
            gt_ligand_file,
            ligand_file,
        ):
            result = {}
            try:
                lig_fail, complex_fail = bust_complex(
                    protein_file, ligand_code, ligand_file=ligand_file
                )
                result["ligand_pbvalid"] = len(lig_fail) == 0
                result["protein_ligand_pbvalid"] = len(complex_fail) == 0
                side_chain_pbvalidity = bust_side_chains(
                    protein_file, ligand_code, ligand_file=gt_ligand_file
                )
                result["side_chain_pbvalidity"] = side_chain_pbvalidity
                side_chain_clashes = count_clashes(
                    protein_file, ligand_code, ligand_file=gt_ligand_file
                )
                result["side_chain_clashes"] = side_chain_clashes
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                return i, None  # Return the index and None if it failed
            return i, result  # Return the index and result if successful

        results = Parallel(n_jobs=-1)(
            delayed(process_protein)(
                i,
                protein_files[i],
                ligand_codes[i],
                gt_ligand_files[i],
                ligand_files[i],
            )
            for i in tqdm(
                range(len(protein_files)), desc="Calculating plausability metrics"
            )
        )
        values = {i: result for i, result in results if result is not None}
        failed = [i for i, result in results if result is None]

        ligand_pbvalidity = np.mean([values[i]["ligand_pbvalid"] for i in values])
        protein_ligand_pbvalidity = np.mean(
            [values[i]["protein_ligand_pbvalid"] for i in values]
        )
        side_chain_pbvalidity = np.mean(
            [values[i]["side_chain_pbvalidity"] for i in values]
        )
        mean_side_chain_pbvalidity = np.mean(
            [values[i]["side_chain_pbvalidity"] for i in values]
        )
        std_side_chain_pbvalidity = np.std(
            [values[i]["side_chain_pbvalidity"] for i in values]
        )
        none_clashes = len(
            [
                values[i]["side_chain_clashes"]
                for i in values
                if values[i]["side_chain_clashes"] == 0
            ]
        ) / len(values)
        no_clashes = np.mean([values[i]["side_chain_clashes"] for i in values])
        std_clashes = np.std([values[i]["side_chain_clashes"] for i in values])
        overall_plausability = [
            (
                True
                if values[i]["ligand_pbvalid"] == 1
                and values[i]["protein_ligand_pbvalid"] == 1
                and values[i]["side_chain_pbvalidity"] == 1
                and values[i]["side_chain_clashes"] == 0
                else False
            )
            for i in values
        ]
        display_dataset_values(
            csv_file,
            values=[
                ligand_pbvalidity,
                protein_ligand_pbvalidity,
                side_chain_pbvalidity,
                f"{mean_side_chain_pbvalidity} ± {std_side_chain_pbvalidity}",
                none_clashes,
                f"{no_clashes} ± {std_clashes}",
                f"{np.mean(overall_plausability)}",
            ],
            value_names=[
                "Ligand PBValid",
                "Protein-Ligand PBValid",
                "Side Chain PBValidity Pass Rate",
                "Side Chain No of PBValid (mean ± std)",
                "No Side Chain Clashes",
                "Side Chain No of Clashes (mean ± std)",
                "Overall Plausability",
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
                    "ligand_pbvalid": [values[i]["ligand_pbvalid"] for i in values],
                    "protein_ligand_pbvalid": [
                        values[i]["protein_ligand_pbvalid"] for i in values
                    ],
                    "side_chain_pbvalidity": [
                        values[i]["side_chain_pbvalidity"] for i in values
                    ],
                    "side_chain_clashes": [
                        values[i]["side_chain_clashes"] for i in values
                    ],
                    "overall_plausability": overall_plausability,
                }
            )
            output_df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
