from collections import namedtuple
from typing import List

import numpy as np

milestone = namedtuple("Milestone", ["step", "value"])


class SchedulerInterface:
    def get_value(self, step) -> int:
        """Return the value of the scheduled parameter for the given step"""


class LinearScheduler(SchedulerInterface):
    """Linearly interpolate the parameter between reference values at given steps.

    The returned value for steps before the first milestone and after the last milestone
    are respectively the first and last values.
    """

    def __init__(self, milestones: List[milestone]):
        assert len(set([m.step for m in milestones])) == len(
            milestones
        ), f"Received non-unique step values in LinearScheduler initialization: {milestones}"
        milestones.sort(key=lambda m: m.step)
        self.milestone_steps = np.array([m.step for m in milestones])
        self.milestone_values = np.array([m.value for m in milestones])

    def get_value(self, step: int):
        return np.interp(x=step, xp=self.milestone_steps, fp=self.milestone_values)
