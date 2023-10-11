"""
This file register gcc_multienv-v0 environment with CompilerGym
and defines the reward function for it
"""

from pathlib import Path

from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

from compiler_gym.envs.gcc_multienv.datasets import *
import math

GCC_MULTIENV_SERVICE_BINARY: Path = runfiles_path(
    "compiler_gym/envs/gcc_multienv/service/gcc-multienv-service"
)


class SizeRuntimeReward(Reward):
    def __init__(self):
        super().__init__(
            name="size_runtime",
            observation_spaces=[
                "runtime_sec",
                "runtime_percent",
                "size",
                "base_size",
                "base_runtime_sec",
                "base_runtime_percent",
            ],
            default_value=0,
            default_negates_returns=False,
            deterministic=False,
            platform_dependent=True,
        )
        self.RUNTIME_WEIGHT = 0.5

    def reset(self, benchmark: str, observation_view):
        """
        Record information about the initital state
        """
        self.prev_runtime_percent = observation_view['runtime_percent']
        self.prev_runtime_sec = observation_view['runtime_sec']
        self.prev_size = observation_view['size']

    def update(self, action, observations, observation_view):
        """
        Calculate and normalize size and runtime reduction in new state compared to previous state.
        If function runtime took less than 1 percent of overall program runtime we consider
        runtime value to be too noisy and not very important to overall runtime, and do not include it
        in reward calculation. Normalized runtime reduction is multiplied by RUNTIME_WEIGHT<1,
        to promote reducing size (while keeping runtime somewhat constant or improving it)
        """
        size_diff_norm = (self.prev_size - observation_view['size']) / self.prev_size
        if self.prev_runtime_percent < 1 and observation_view['runtime_percent'] < 1:
            runtime_diff_norm = 0
        else:
            if self.prev_runtime_sec == 0:
                runtime_diff_norm = 1
            else:
                runtime_diff_norm = (self.prev_runtime_sec - observation_view['runtime_sec']) / self.prev_runtime_sec

        diff = size_diff_norm + runtime_diff_norm * self.RUNTIME_WEIGHT

        self.prev_runtime_percent = observation_view['runtime_percent']
        self.prev_runtime_sec = observation_view['runtime_sec']
        self.prev_size = observation_view['size']

        return diff


register(
    id="gcc_multienv-v0",
    entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
    kwargs={
        "service": GCC_MULTIENV_SERVICE_BINARY,
        "rewards": [SizeRuntimeReward()],
        "datasets": [MultienvDataset()],
    },
)
