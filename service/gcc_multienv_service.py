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
from compiler_gym.envs.gcc_multienv.shuffler import *
import os
import re

class GccMultienvCompilationSession(CompilationSession):

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
        self.target_lists = [int(x) for x in self.parsed_bench.params.get("list", ['1', '2', '3'])]
        self._lists_valid = False
        if 3 not in self.target_lists:
            self._lists_valid = True
        self._binary_valid = False
        self._src_copied = False
        self._wd_valid = False
        self.copy_bench()
        self.prep_wd()
        self.runtime = None
        self.size = None
        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        action_string = action.string_value
        if action_string == None:
            raise ValueError("Expected pass name, got None")
        logging.info("Applying action %s", action_string)

        if re.match("none_pass", action_string[1:] if action_string[0] == '>' else action_string) != None:
            return False, False, False

        regex_res = re.search("\?(\d)", action_string)
        if regex_res == None:
            if (len(self.target_lists) > 1) or (self.target_lists == [0]):
                raise ValueError("Expected specified target list in pass arg")
            else:
                pass_list = self.target_lists[0]
        else:
            pass_list = int(regex_res.group(1))
            action_string = re.match("(.*)\?", action_string).group(1)
            if pass_list not in self.target_lists:
                action_string = "\n" + action_string

        list_num = get_pass_list(self.actions_lib, action_string[1:] if action_string[0] == '>' else action_string)
        if list_num == -1:
            raise ValueError(f"Unknown pass {action_string}")

        with open(self.working_dir.joinpath(f"bench/list{pass_list}.txt"), "a") as pass_file:
            pass_file.write(action_string + "\n")

        self._lists_valid = True
        for elem in self.target_lists:
            if valid_pass_seq(self.actions_lib, self.get_list(elem), elem) != 0:
                self._lists_valid = False

        new_list = get_action_list(self.actions_lib, [], self.get_list(pass_list), pass_list)
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
        self.size = None
        self.runtime = None
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
            copytree(self.parsed_bench.path, self.working_dir.joinpath('bench'), dirs_exist_ok=True)
            if "bench_repeats" in self.parsed_bench.params:
                with open(self.working_dir.joinpath("bench/_finfo_dataset"), "w") as f:
                    f.write(self.parsed_bench.params["bench_repeats"][0])
            self._src_copied = True

    def prep_wd(self):
        if not self._wd_valid:
            copy2("../shuffler/lists/to_shuffle1.txt", self.working_dir.joinpath('bench/list1.txt'))
            copy2("../shuffler/lists/to_shuffle2.txt", self.working_dir.joinpath('bench/list2.txt'))
            copy2("../shuffler/lists/to_shuffle3.txt", self.working_dir.joinpath('bench/list3.txt'))
            for elem in self.target_lists:
                os.remove(self.working_dir.joinpath(f'bench/list{elem}.txt'))
            call('touch list1.txt list2.txt list3.txt', shell=True, cwd=self.working_dir.joinpath('bench'))
            self._wd_valid = True

    def get_baseline(self):
        if ("base_size" in self.parsed_bench.params) or ("base_runtime" in self.parsed_bench.params):
            self.baseline_size = int(self.parsed_bench.params["base_size"][0])
            self.baseline_runtime = float(self.parsed_bench.params["base_runtime"][0])
        else:
            cache_lists_valid = self._lists_valid
            self.compile_baseline()
            self.baseline_size = int(self.parsed_bench.params.get("base_size", [self.get_size()])[0])
            self.baseline_runtime = float(self.parsed_bench.params.get("base_runtime", [self.get_runtime()])[0])
            self._binary_valid = False
            self._lists_valid = cache_lists_valid

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
        if self.runtime == None:
            arg = " ".join(self.parsed_bench.params.get("run"))
            start_time = clock_gettime(CLOCK_MONOTONIC)
            run(f'qemu-aarch64 -L /usr/aarch64-linux-gnu ./bench.elf {arg}', shell=True, cwd=self.working_dir.joinpath('bench'), check=True)
            end_time = clock_gettime(CLOCK_MONOTONIC)
            self.runtime = end_time - start_time
        return self.runtime

    def get_size(self):
        if not self._lists_valid:
            return 0
        if not self._binary_valid:
            self.compile()
        if self.size == None:
            self.size = int(run('size bench.elf', shell=True, capture_output=True, cwd=self.working_dir.joinpath('bench')).stdout.split()[6])
        return self.size

    def get_passes(self):
        passes = []
        for i in self.target_lists:
            with open(self.working_dir.joinpath(f"bench/list{i}.txt"), "r") as pass_file:
                passes += pass_file.read().splitlines()
        return passes

    def get_list(self, list_num):
        with open(self.working_dir.joinpath(f"bench/list{list_num}.txt"), "r") as pass_file:
            passes = pass_file.read().splitlines()
        return passes

if __name__ == "__main__":
    create_and_run_compiler_gym_service(GccMultienvCompilationSession)


