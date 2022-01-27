import click
import os
import shutil
import subprocess
from configparser import ConfigParser
from pathlib import Path


@click.command()
@click.argument("name", type=str)
@click.argument("template_dir", type=click.Path(exists=True))
@click.option("--database", type=click.Path(exists=True), required=True)
@click.option("--featname", type=str, required=True)
@click.option("--overwrite", is_flag=True)
def create(name, template_dir, database, featname, overwrite):

    from assist.tools import read_config

    template_dir, database = map(Path, (template_dir, database))
    config_dir = template_dir.parent/name
    if config_dir.exists() and not overwrite:
        raise click.BadParameter(f"{config_dir} exists.")
    elif config_dir.exists():
        shutil.rmtree(config_dir)
    click.echo(f"Copy {template_dir} to {config_dir}")
    shutil.copytree(template_dir, config_dir, symlinks=True)
    featcfg = read_config(config_dir/"features.cfg")
    featcfg["features"]["file"] = str(database)
    click.echo("Saving feature config")
    with open(config_dir/"features.cfg", "w") as f:
        featcfg.write(f)
    datacfg = read_config(config_dir/"database.cfg")
    to_replace = None
    for section in datacfg.sections():
        featfile = datacfg[section]["features"]
        if to_replace is None:
            to_replace = Path(featfile ).parent.name
            click.echo(f"Replace {to_replace} with {featname}")
        datacfg[section]["features"] = featfile.replace(to_replace, featname)
    click.echo("Saving database config")
    with open(config_dir/"database.cfg", "w") as f:
        datacfg.write(f)


if __name__ == "__main__":
    create(obj={})
