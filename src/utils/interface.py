from rich.console import Console
from rich.panel import Panel
from typing import List, Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


def format_value(val):
    """Format the value to 3 significant figures.

    Args:
        val (Any): The value to format.
    """
    if isinstance(val, (int, float)):
        return f"{val:.3g}"
    return str(val)


def display_single_value(
    protein_file,
    ligand_code,
    ligand_file,
    value: Any | List,
    value_name: Any | List = "Value",
    title: str = "Pocket Metric",
):
    """
    Display the value for a plausibility metric in a table format with separated protein and ligand sections.

    Args:
        protein_file (str): The path to the protein file.
        ligand_code (str): The ligand code.
        ligand_file (str): The path to the ligand file.
        value (float | List[float]): The value or list of values.
        value_name (str | List[str]): The name of the value or list of value names.
        title (str): The title of the table.
    """
    console = Console()

    # Create a table to display the content
    table = Table(
        title=title + f" - ../{'/'.join(protein_file.split('/')[-4:])}",
        show_header=True,
        header_style="bold blue",
    )

    # Add columns to the table
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Add rows for the values
    if isinstance(value, list):
        assert isinstance(
            value_name, list
        ), "value_name must be a list if value is a list"
        for i, val in enumerate(value):
            table.add_row(str(value_name[i]), str(format_value(val)))
    else:
        table.add_row(str(value_name), str(format_value(value)))

    console.print(table)


def display_dataset_values(
    csv_file: str,
    values: List[Any],
    value_names: List[str],
    title: str = "Pocket Metric",
):
    """
    Display the values for a plausibility metric in a table format.

    Args:
        csv_file (str): The path to the CSV file.
        values (List[float]): The values.
        value_names (List[str]): The names of the values.
        title (str): The title of the table.
    """
    console = Console()

    # Create a table to display the content
    table = Table(
        title=title + f" - ../{'/'.join(csv_file.split('/')[-4:])}",
        show_header=True,
        header_style="bold blue",
    )

    # Add columns to the table
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    def format_value(val):
        """Format the value to 3 significant figures."""
        if isinstance(val, (int, float)):
            return f"{val:.3g}"  # Format to 3 significant figures
        return str(val)  # If it's not a number, return as a string

    # Add rows for the values
    for i, val in enumerate(values):
        table.add_row(str(value_names[i]), format_value(val))

    console.print(table)
