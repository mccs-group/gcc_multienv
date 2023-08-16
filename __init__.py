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
        self.base_runtime = None
        self.base_size = None
        self.prev_state_value = None
        self.prev_runtime_percent = None

    def reset(self, benchmark: str, observation_view):
        self.base_runtime = observation_view["base_runtime_sec"]
        self.base_size = observation_view["base_size"]
        self.prev_runtime_percent = observation_view["runtime_percent"]
        self.prev_state_value = self.state_value(observation_view)

    def state_value(self, observation_view):
        size_normalized = (self.base_size - observation_view["size"]) / self.base_size

        if observation_view["runtime_percent"] <= 5.0:
            runtime_normalized = 0.0
        else:
            runtime_normalized = (
                self.base_runtime - observation_view["runtime_sec"]
            ) / self.base_runtime

        if size_normalized < 0:
            return size_normalized
        else:
            if runtime_normalized > 0:
                return size_normalized
            else:
                return size_normalized - math.exp(-runtime_normalized)

    def update(self, action, observations, observation_view):
        new_state_value = self.state_value(observation_view)
        diff = new_state_value - self.prev_state_value
        self.prev_state_value = new_state_value
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
