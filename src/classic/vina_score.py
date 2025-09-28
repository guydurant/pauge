import os
from tempfile import NamedTemporaryFile
import subprocess
import requests


def check_smina():
    # Check if the Smina executable already exists
    if not os.path.exists("smina.static"):
        raise FileNotFoundError(
            "The Smina executable was not found. Please download it from https://sourceforge.net/projects/smina/files/smina.static"
        )


def download_smina():
    if not os.path.exists("smina.static"):
        url = "https://sourceforge.net/projects/smina/files/smina.static"
        r = requests
        r = requests.get(url)
        with open("smina.static", "wb") as f:
            f.write(r.content)
    os.system("chmod +x smina.static")


def vina_score(generated_prot_file, ligand_code, ligand_file=None):
    """Compute the Vina score for a generated protein pocket.

    Args:
        generated_prot_file (str): The path to the generated protein file.
        ligand_code (str): The ligand code.
        ligand_file (str): The path to the ligand file.

    Returns:
        float: The Vina score.
    """
    download_smina()

    if not ligand_file:
        raise ValueError(
            "The ligand file must be provided. Have not implemented using ligand code yet."
        )

    temp_file = NamedTemporaryFile(delete=False, suffix=".pdb")
    cleaned_protein = "\n".join(
        [
            line
            for line in open(generated_prot_file, "r").read().splitlines()
            if line.startswith("ATOM")
        ]
    )
    temp_file.write(cleaned_protein.encode("utf-8"))
    temp_file.close()

    command = [
        "./smina.static",
        "--receptor",
        temp_file.name,
        "--ligand",
        ligand_file,
        "--score_only",
        "--cpu",
        "10",
    ]
    output = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    text = output.stdout.split()
    # get index for Affinity
    index = text.index("Affinity:")
    return float(text[index + 1])


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

        value = vina_score(generated_prot_file, ligand_code, ligand_file=ligand_file)
        display_single_value(
            generated_prot_file,
            ligand_code,
            ligand_file,
            value,
            value_name="Vina Score (kcal/mol)",
        )

    main()
