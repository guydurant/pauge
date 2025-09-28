from Bio import PDB
import io
import numpy as np
from src.utils.structure import get_molblocks

AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def aarecovery_from_pdbblocks(pdbblock_gen, full_pdbblock_gt, pdbblock_gt):
    gen_residues = {}
    gen_keys = []
    gt_residues = {}
    gt_keys = []
    for line in pdbblock_gen.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # also include
            if line[17:20].strip() not in AA:
                continue
            key = (line[22:26].strip(), line[21].strip(), line[26].strip())
            if key not in gen_keys:
                gen_keys.append(key)
            gen_residues[gen_keys.index(key)] = line[17:20].strip()
            # gen_residues[key] = line[17:20].strip()
    for line in full_pdbblock_gt.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if line[17:20].strip() not in AA:
                continue
            key = (line[22:26].strip(), line[21].strip(), line[26].strip())
            if key not in gt_keys:
                gt_keys.append(key)
            # gt_residues[gt_keys.index(key)] = line[17:20].strip()
            # gt_residues[key] = line[17:20].strip()
    for line in pdbblock_gt.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if line[17:20].strip() not in AA:
                continue
            key = (line[22:26].strip(), line[21].strip(), line[26].strip())
            gt_residues[gt_keys.index(key)] = line[17:20].strip()
            # gt_residues[key] = line[17:20].strip()
    assert len(gen_keys) == len(
        gt_keys
    ), f"Different number of residues.{len(gen_keys)} vs {len(gt_keys)}, {set(gen_keys) - set(gt_keys)}"
    # print(gen_keys, gt_keys)
    # print(gen_residues, gt_residues)
    match = sum(
        1
        for key in gen_residues
        if key in gt_residues and gen_residues[key] == gt_residues[key]
    )

    return match / len(gt_residues)


def aarecovery(
    generated_prot_file,
    ground_truth_prot_file,
    ground_truth_lig_file,
    ligand_code,
    cutoff=5,
    ligand_file=None,
):
    prot_gen_block, lig_gen_block = get_molblocks(
        generated_prot_file,
        ligand_code,
        -1,
        ligand_file,
    )
    prot_gt_block, lig_gt_block = get_molblocks(
        ground_truth_prot_file,
        ligand_code,
        cutoff,
        ground_truth_lig_file,
    )
    full_prot_gt_block, full_lig_gt_block = get_molblocks(
        ground_truth_prot_file,
        ligand_code,
        -1,
        ground_truth_lig_file,
    )
    return aarecovery_from_pdbblocks(prot_gen_block, full_prot_gt_block, prot_gt_block)


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

        value = aarecovery(
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
            value_name="Amino Acid Recovery",
        )

    main()
