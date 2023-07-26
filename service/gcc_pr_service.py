#! /usr/bin/python3
import logging
from pathlib import Path
from typing import Optional, Tuple
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    ActionSpace,
    NamedDiscreteSpace,
    StringSequenceSpace,
    Event,
    Space,
    ObservationSpace,
    DoubleRange,
    Int64Range,
    ListEvent,
    )
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from shutil import (copytree, copy2)
from compiler_gym.datasets import (BenchmarkUri, Benchmark)
from subprocess import *
from time import *
from compiler_gym.envs.gcc_pr.shuffler import *
import os

class GccPRCompilationSession(CompilationSession):

    compiler_version: str = "7.3.0"

    actions_lib = setuplib("../shuffler/libactions.so")
    action_list1 = get_action_list(actions_lib, [], [], 1)
    action_list2 = get_action_list(actions_lib, [], [], 2)
    action_list3 = get_action_list(actions_lib, [], [], 3)

    action_spaces = [
        ActionSpace(
            name="list_all",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=action_list1 + action_list2 + action_list3
                    ),
                ),
            ),
        ActionSpace(
            name="list1",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=action_list1
                    ),
                ),
            ),
        ActionSpace(
            name="list2",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=action_list2
                    ),
                ),
            ),
        ActionSpace(
            name="list3",
            space=Space(
                named_discrete=NamedDiscreteSpace(
                    name=action_list3
                    ),
                ),
            ),
        ]

    observation_spaces = [
        ObservationSpace(
            name="runtime",
            space=Space(
                double_value=DoubleRange(min=0),
                ),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
                ),
            ),
        ObservationSpace(
            name="passes",
            space=Space(
                string_sequence=StringSequenceSpace(length_range=Int64Range(min=0)),
                ),
            deterministic=True,
            platform_dependent=False,
            default_observation=Event(
                string_value="",
                ),
            ),
        ObservationSpace(
            name="size",
            space=Space(
                int64_value=Int64Range(min=0),
                ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                int64_value=0,
                ),
            ),
        ObservationSpace(
            name="base_runtime",
            space=Space(
                double_value=DoubleRange(min=0),
                ),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
                ),
            ),
        ObservationSpace(
            name="base_size",
            space=Space(
                int64_value=Int64Range(min=0),
                ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                int64_value=0,
                ),
            ),
        ]

    def __init__(self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark):
        super().__init__(working_directory, action_space, benchmark)
        self.parsed_bench = BenchmarkUri.from_string(benchmark.uri)
        self.baseline_size = None
        self.baseline_runtime = None
        self.target_list = int(self.parsed_bench.params.get("list", ['0'])[0])
        self._lists_valid = False
        self._binary_valid = False
        self._src_copied = False
        self._wd_valid = False
        self.copy_bench()
        self.prep_wd()
        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        action_string = action.string_value
        if action_string == None:
            raise ValueError("Expected pass name, got None")
        logging.info("Applying action %s", action_string)

        if self.target_list == 0:
            raise NotImplementedError()

        list_num = get_pass_list(self.actions_lib, action_string)
        if list_num == -1:
            raise ValueError(f"Unknown pass {action_string}")
#         if (self.target_list != 0) && (list_num != self.target_list):
#             raise ValueError(f"Pass {action_string} from incorrect list ({list_num} vs {self.target_list})")

        with open(self.working_dir.joinpath(f"bench/list{self.target_list}.txt"), "a") as pass_file:
            pass_file.write(action_string + "\n")

        list_check = valid_pass_seq(self.actions_lib, self.get_passes(), self.target_list)
        if list_check == 0:
            self._lists_valid = True
        else:
            self._lists_valid = False

        new_list = get_action_list(self.actions_lib, [], self.get_list(self.target_list), self.target_list)
        if new_list != []:
            new_space = ActionSpace(
                name="new_space",
                space=Space(
                    named_discrete=NamedDiscreteSpace(
                        name=new_list
                        ),
                    ),
                )
        else:
            new_space = None

        self._binary_valid = False
        return True if new_space == None else False, new_space, False

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info("Computing observation from space %s", observation_space.name)
        if observation_space.name == "runtime":
            return Event(double_value=self.get_runtime())
        elif observation_space.name == "size":
            return Event(int64_value=self.get_size())
        elif observation_space.name == "base_runtime":
            if self.baseline_runtime == None:
                self.get_baseline()
            return Event(double_value=self.baseline_runtime)
        elif observation_space.name == "base_size":
            if self.baseline_size == None:
                self.get_baseline()
            return Event(int64_value=self.baseline_size)
        elif observation_space.name == "passes":
            return Event(event_list=ListEvent(event=list(map(lambda name: Event(string_value=name), self.get_passes()))))
        else:
            raise KeyError(observation_space.name)

    def copy_bench(self):
        if not self._src_copied:
            print(self.working_dir)
            copytree(self.parsed_bench.path, self.working_dir.joinpath('bench'), dirs_exist_ok=True)
            self._src_copied = True

    def prep_wd(self):
        if not self._wd_valid:
            if self.target_list != 0:
                copy2("../shuffler/lists/to_shuffle1.txt", self.working_dir.joinpath('bench/list1.txt'))
                copy2("../shuffler/lists/to_shuffle2.txt", self.working_dir.joinpath('bench/list2.txt'))
                copy2("../shuffler/lists/to_shuffle3.txt", self.working_dir.joinpath('bench/list3.txt'))
                os.remove(self.working_dir.joinpath(f'bench/list{self.target_list}.txt'))
            call('touch list1.txt list2.txt list3.txt', shell=True, cwd=self.working_dir.joinpath('bench'))
            self._wd_valid = True

    def get_baseline(self):
        self.compile_baseline()
        self.baseline_size = self.get_size()
        self.baseline_runtime = self.get_runtime()
        self._binary_valid = False
        self._lists_valid = False

    def compile_baseline(self):
        base_opt = " ".join(self.parsed_bench.params.get("base_opt", ["-O2"]))
        src_dir = " ".join(self.parsed_bench.params.get("src_dir"))
        build_arg = " ".join(self.parsed_bench.params.get("build"))
        check_call(f'''$AARCH_GCC {base_opt} {build_arg} {src_dir}*.c -o bench.elf''', shell=True, cwd=self.working_dir.joinpath('bench'))
        self._binary_valid = True
        self._lists_valid = True

    def compile(self):
        src_dir = " ".join(self.parsed_bench.params.get("src_dir"))
        build_arg = " ".join(self.parsed_bench.params.get("build"))
        plugin_args = "-fplugin-arg-plugin-pass_file=list1.txt -fplugin-arg-plugin-pass_file=list2.txt -fplugin-arg-plugin-pass_file=list3.txt"
        check_call(f'''$AARCH_GCC -O2 -fplugin=$GCC_PLUGIN -fplugin-arg-plugin-pass_replace {plugin_args} {build_arg} {src_dir}*.c -o bench.elf''', shell=True, cwd=self.working_dir.joinpath('bench'))
        self._binary_valid = True

    def get_runtime(self):
        if not self._lists_valid:
            return 0
        if not self._binary_valid:
            self.compile()
        arg = " ".join(self.parsed_bench.params.get("run"))
        start_time = clock_gettime(CLOCK_MONOTONIC)
        run(f'qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench.elf {arg}', shell=True, cwd=self.working_dir.joinpath('bench'), check=True)
        end_time = clock_gettime(CLOCK_MONOTONIC)
        return end_time - start_time

    def get_size(self):
        if not self._lists_valid:
            return 0
        if not self._binary_valid:
            self.compile()
        return int(run('size bench.elf', shell=True, capture_output=True, cwd=self.working_dir.joinpath('bench')).stdout.split()[6])

    def get_passes(self):
        passes = []
        if self.target_list == 0:
            for i in range(1, 4):
                with open(self.working_dir.joinpath(f"bench/list{i}.txt"), "r") as pass_file:
                    passes += pass_file.read().splitlines()
        else:
            passes += self.get_list(self.target_list)
        return passes

    def get_list(self, list_num):
        with open(self.working_dir.joinpath(f"bench/list{list_num}.txt"), "r") as pass_file:
            passes = pass_file.read().splitlines()
        return passes


if __name__ == "__main__":
    create_and_run_compiler_gym_service(GccPRCompilationSession)


