import click

from .recorders import HandGestureRecorder

labels = ["point"]


@click.command()
@click.option(
    "-l",
    "--label",
    help="Label to use.",
    required=True,
    type=click.Choice(labels, case_sensitive=True),
)
def main(label):
    with HandGestureRecorder(label) as recorder:
        recorder.run()
