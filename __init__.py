from pathlib import Path

from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

from compiler_gym.envs.gcc_multienv.datasets import *
import math

GCC_MULTIENV_SERVICE_BINARY: Path = runfiles_path("compiler_gym/envs/gcc_multienv/service/gcc-multienv-service")

class SizeRuntimeReward(Reward):

    def __init__(self):
        super().__init__(
                name="size_runtime",
                observation_spaces=["runtime", "size", "base_size", "base_runtime"],
                default_value=0,
                default_negates_returns=False,
                deterministic=False,
                platform_dependent=True,
        )
        self.base_runtime = None
        self.base_size = None

    def reset(self, benchmark: str, observation_view):
        self.base_runtime = observation_view["base_runtime"]
        self.base_size = observation_view["base_size"]

    def update(self, action, observations, observation_view):
        del action
        size = observation_view["size"]
        if size == 0:
            return 0
        size_delta = (self.base_size - size) / self.base_size
        if size_delta < 0:
            return size_delta
        else:
            runtime = observation_view["runtime"]
            runtime_delta = (self.base_runtime - runtime) / self.base_runtime
            if runtime_delta >= 0:
                return size_delta
            else:
                return size_delta + runtime_delta
        return size_delta

register(
        id="gcc_multienv-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": GCC_MULTIENV_SERVICE_BINARY,
            "rewards": [SizeRuntimeReward()],
            "datasets": [MultienvDataset()],
        },
)
