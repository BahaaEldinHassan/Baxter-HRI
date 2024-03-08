import click

from .inferrers import HandGestureInferrerLive
from .recorders import HandGestureRecorderLive, BodyGestureRecorderLive
from .settings import LABELS
from .experiments import YoloRecorderLive


@click.group()
def command_group():
    ...


@command_group.command()
def infer():
    with HandGestureInferrerLive() as inferrer:
        inferrer.run()

RECORDERS = {
    "body": BodyGestureRecorderLive,
    "hand": HandGestureRecorderLive,
    "yolo": YoloRecorderLive,
}

@command_group.command()
@click.option(
    "-l",
    "--label",
    help="Label to use.",
    required=True,
    type=click.Choice(LABELS, case_sensitive=True),
)
@click.option(
    "-r",
    "--recorder",
    help="Recorder to use.",
    required=True,
    type=click.Choice(list(RECORDERS.keys()), case_sensitive=True),
)
def record(label, recorder):
    recorder_cls = RECORDERS[recorder]

    with recorder_cls(label) as recorder:
        recorder.run()
