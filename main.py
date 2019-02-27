import click

from src.train import train


# Create a nested command from command groups in the src package
@click.group()
def cli():
    pass


# We must use add_command instead of CommandCollection to get a nested structure.
# https://stackoverflow.com/a/39416589
cli.add_command(train)


# Initialize the command line interface
if __name__ == '__main__':
    cli()
