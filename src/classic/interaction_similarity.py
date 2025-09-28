from src.utils.interactions import get_interactions, prepare_plipobj

from warnings import filterwarnings

filterwarnings("ignore")


def interaction_similarity(
    generated_prot_file,
    ground_truth_prot_file,
    ground_truth_lig_file,
    ligand_code,
    cutoff=5,
    ligand_file=None,
):
    gen_plipobj = prepare_plipobj(generated_prot_file, ligand_file, None, False)
    # get last atom of protein
    gen_interactions = get_interactions(gen_plipobj)

    gt_plipobj = prepare_plipobj(
        ground_truth_prot_file, ground_truth_lig_file, None, False
    )
    gt_interactions = get_interactions(gt_plipobj)
    gen_interactions = gen_interactions[
        [
            "type",
            "ligand_id",
        ]
    ]
    # drop hydrophobic interacitions
    gen_interactions = gen_interactions[gen_interactions["type"] != "hydrophobic"]
    if len(gen_interactions) == 0:
        return 0
    gt_interactions = gt_interactions[
        [
            "type",
            "ligand_id",
        ]
    ]
    if len(gt_interactions) == 0:
        return -1
    gt_interactions = gt_interactions[gt_interactions["type"] != "hydrophobic"]
    gt_interactions_copy = gt_interactions.copy()
    correct = 0
    for index, gen_interaction in gen_interactions.iterrows():
        # Check if gen_interaction exists in gt_interactions_copy
        match = gt_interactions_copy[
            (gt_interactions_copy == gen_interaction).all(axis=1)
        ]
        if not match.empty:
            # Remove the first matched interaction from gt_interactions_copy
            gt_interactions_copy = gt_interactions_copy.drop(match.index[0])
            correct += 1
    # print(correct / (len(gt_interactions) + len(gen_interactions) - correct))
    return correct / (len(gt_interactions) + len(gen_interactions) - correct)


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
    @click.option(
        "--ground_truth_prot_file",
        "-t",
        required=True,
        type=str,
        help="Path to the ground truth protein file.",
    )
    @click.option(
        "--ground_truth_lig_file",
        "-p",
        required=True,
        type=str,
        help="Path to the ground truth ligand file.",
    )
    def main(
        generated_prot_file,
        ligand_code,
        ligand_file,
        ground_truth_prot_file,
        ground_truth_lig_file,
    ):
        """Command line interface for the Amino Acid Recovery metric."""
        if not ligand_code and not (ligand_file and ground_truth_lig_file):
            raise click.UsageError(
                "Only--ligand_code or both --ligand_file and --ground_truth_lig_file must be provided."
            )

        value = interaction_similarity(
            generated_prot_file,
            ground_truth_prot_file,
            ground_truth_lig_file,
            ligand_code,
            ligand_file=ligand_file,
        )
        print(value)
        print(type(value))
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value,
            value_name="Interaction Similarity",
        )

    main()
