from pymol import cmd

COMMAND = "source /data/localhost/not-backed-up/durant/miniconda3/bin/activate && conda activate {env_name} && cd {model_dir} && python {script} {options}"


def make_complex(protein_file, ligand_file, out_file):
    cmd.load(protein_file, "prot")
    cmd.load(ligand_file, "lig")
    cmd.save(out_file, "prot or lig")
    cmd.delete("all")
    return out_file


def ligandmpnn(prot_file, lig_file, out_folder):
    return COMMAND.format(
        env_name="ligandmpnn_env",
        model_dir="/vols/opig/projects/guy-affinity/POCKET_DESIGN/LigandMPNN",
        script="run_pocket_gen.py",
        options=f'--seed 1 --pdb_path {prot_file} --ligand_file {lig_file}  --out_folder {out_folder} --model_type "ligand_mpnn" --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_005_25.pt" --pack_with_ligand_context 1 --pack_side_chains 1 --number_of_packs_per_design 1',
    )


def pocketgen(prot_file, lig_file, out_folder, minimise=True):
    return COMMAND.format(
        env_name="pocketgen",
        model_dir="/vols/opig/projects/guy-affinity/POCKET_DESIGN/PocketGen",
        script="inference.py",
        options=(
            f"--protein_file {prot_file} --ligand_file {lig_file}  --outdir {out_folder} --minimise"
            if minimise
            else f"--complex_file {complex_file}  --outdir {out_folder}"
        ),
    )


def flowsite(prot_file, lig_file, out_folder):
    command = COMMAND.format(
        env_name="flowsite",
        model_dir="/vols/opig/users/durant/inverse_folding_ligands/FlowSite",
        script="-m inference",
        options=f"--num_inference 10 --out_dir {out_folder} --design_residues $design_res --ligand {lig_file} --protein {prot_file} --batch_size 16 --pocket_def_ligand {lig_file} --checkpoint pocket_gen/lf5t55w4/checkpoints/best.ckpt --save_inference --run_test --run_name inference1 --layer_norm --fake_constant_dur 100000 --fake_decay_dur 10 --fake_ratio_start 0.2 --fake_ratio_end 0.2 --residue_loss_weight 0.2 --use_tfn --use_inv --time_condition_tfn --correct_time_condition --time_condition_inv --time_condition_repeat --flow_matching --flow_matching_sigma 0.5 --prior_scale 1 --num_angle_pred 5 --ns 48 --nv 12 --batch_norm --self_condition_x --self_condition_inv --no_tfn_self_condition_inv --self_fancy_init --pocket_residue_cutoff 8",
    )
    commands = command.split("&&")
    prep = f"python prepare.py {prot_file} {lig_file} {out_folder} && design_res=$(cat {out_folder}/pocket_residues.txt)"
    commands = commands[:3] + [prep] + commands[3:]
    print(commands)
    return " && ".join(commands)


commands = {
    "ligandmpnn": ligandmpnn,
    "pocketgen": pocketgen,
    "flowsite": flowsite,
}
