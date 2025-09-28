<h1 align="center">Pauge</h1>

Pauge is a tool to assess the quality of designed ligand protein binding pockets. 

These assessments are divided into two catagories: classic metrics and plausibility metrics. Classic metrics are metrics that have been used in the field for a long time, such as amino acid recovery and binding energy using Vina/Smina. Plausibility metrics are metrics that assess the plausibility of the designed pocket, such as side chain clashes and PoseBusters.

## Installation

Pauge can be installed using pip after cloning the repository:

```bash
pip install -e .
```

## Usage
Pauge can be run from the command line in `classic` and `plausibility` modes.

```bash
pauge_classic --help
pauge_plausability --help
```

## Metrics
Pauge currently supports the following metrics:
- Classic Metrics:
  - Amino Acid Recovery
  - Binding Energy (Vina/Smina)
  - Interaction Recovery (PLIP)
  - Missed Hydrogen Bonds (PLIP)

- Plausibility Metrics:
    - Side Chain Clashes
    - Side Chain PBValidity
    - Ligand PBValidity
    - Protein-Ligand PBValidity

## Mutatability

This additional tool assesses how responsive tools are to changes in ligand atoms for the local pocket prediction. It is explained further in my thesis. Requires installation of the specific methods to work and is a bit more involved to run.

```bash
python mutatability.mutate --help
python mutatability.sequence --help
```
`mutate` takes predictions and alters the ligand atom, and repredicts the pocket using the new ligand. `sequence` takes the predictions and calculates the sequence recovery of for local positions.