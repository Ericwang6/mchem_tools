"""CLI entry point: convert PDB to SQLite DB, solvate PDB, etc."""

import os

import glob

import rich_click as click

from mchem.forcefield import ForceField
from mchem.fileformats import load_pdb, read_pdb_box, write_pdb
from mchem.system import Box
from mchem.solvate import solvate
from mchem.template import loadNamedTemplateDefinitions
from mchem.trjconv import trjconv


@click.group()
def main() -> None:
    """M-Chem Front-end: PDB conversion and solvation tools."""
    pass


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=str),
    help="Input PDB file.",
)
@click.option(
    "-f",
    "--forcefield",
    default="amoebabio18.xml",
    type=click.Path(path_type=str),
    help="Force field XML file.",
)
@click.option(
    "--amber",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Use AMBER-style residue templates",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=str),
    help="Output SQLite DB file.",
)
def convert(input_path: str, amber: bool, forcefield: str, output_path: str) -> None:
    """Convert PDB to SQLite-DB formatted force-field parameters."""

    loadNamedTemplateDefinitions("amber" if amber else "amoeba")
    top = load_pdb(input_path)
    ff = ForceField(forcefield)
    system = ff.createSystem(top)
    box = read_pdb_box(input_path)
    if box is not None:
        system.addTerm(Box(*box), "box")
    system.save(output_path)


@main.command("solvate")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=str),
    help="Input PDB file (solute only, no water).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=str),
    help="Output solvated PDB file.",
)
@click.option(
    "--box-shape",
    "box_shape",
    type=click.Choice(["cube", "dodecahedron", "octahedron"], case_sensitive=False),
    default="cube",
    help="Periodic box shape. Default: cube.",
)
@click.option(
    "--buffer",
    type=float,
    default=10.0,
    help="Minimum padding (Angstroms) between solute and box edge. Default: 10.0.",
)
@click.option(
    "--no-neutralize",
    "neutralize",
    is_flag=True,
    default=True,
    flag_value=False,
    help="Do not add counterions to neutralize the system (default is to neutralize).",
)
@click.option(
    "--ionic-strength",
    type=float,
    default=0.0,
    help="Ionic strength in mol/L (molar). Default: 0.0.",
)
@click.option(
    "--positive-ion",
    default="Na+",
    help="Positive ion type (e.g. Na+, K+). Default: Na+.",
)
@click.option(
    "--negative-ion",
    default="Cl-",
    help="Negative ion type (e.g. Cl-). Default: Cl-.",
)
@click.option(
    "--amber",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Use AMBER-style residue templates",
)
def solvate_cmd(
    input_path: str,
    output_path: str,
    box_shape: str,
    buffer: float,
    neutralize: bool,
    ionic_strength: float,
    positive_ion: str,
    negative_ion: str,
    amber: bool,
) -> None:
    """Solvate a solute PDB with water and ions in a periodic box."""
    loadNamedTemplateDefinitions("amber" if amber else "amoeba")
    top = load_pdb(input_path)
    positions = top.coordinates
    solv_top, solv_pos, box_vectors = solvate(
        top,
        positions,
        box_shape=box_shape,
        buffer=buffer,
        neutralize=neutralize,
        ionic_strength=ionic_strength,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
    )
    write_pdb(output_path, solv_top, solv_pos, box_vectors=box_vectors)
    click.echo(f"Solvated PDB written to {output_path}")


@main.command("trjconv")
@click.option(
    "-p",
    "--topology",
    "topology_path",
    required=True,
    type=click.Path(exists=True, path_type=str),
    help="Topology file (currently .db).",
)
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=str),
    help="Input M-Chem HDF5 trajectory (.h5).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=str),
    help="Output structure file (currently .pdb).",
)
@click.option(
    "-f",
    "--frame",
    default=-1,
    type=int,
    show_default=True,
    help="Frame to extract (native 1-based index; -1 selects the last frame).",
)
def trjconv_cmd(
    topology_path: str, input_path: str, output_path: str, frame: int
) -> None:
    """Convert an M-Chem trajectory frame to a PDB file."""
    written = trjconv(topology_path, input_path, output_path, frame=frame)
    click.echo(f"Wrote frame {written.index} to {output_path}")


if __name__ == "__main__":
    main()
