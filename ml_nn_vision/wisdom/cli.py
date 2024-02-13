import click

from .inferrers import HandGestureInferrer
from .recorders import HandGestureRecorder
from .settings import LABELS


@click.group()
def command_group():
    ...


@command_group.command()
def infer():
    with HandGestureInferrer() as inferrer:
        inferrer.run()


@command_group.command()
@click.option(
    "-l",
    "--label",
    help="Label to use.",
    required=True,
    type=click.Choice(LABELS, case_sensitive=True),
)
def record(label):
    with HandGestureRecorder(label) as recorder:
        recorder.run()