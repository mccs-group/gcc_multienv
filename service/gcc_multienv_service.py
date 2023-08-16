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
    ByteSequenceSpace,
    ByteTensor,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from shutil import copytree, copy2, rmtree
from compiler_gym.datasets import BenchmarkUri, Benchmark
from subprocess import *
from time import *
from compiler_gym.envs.gcc_multienv.shuffler import *
import os, sys
import re
import socket
import errno
import struct


class GccMultienvCompilationSession(CompilationSession):
    compiler_version: str = "7.3.0"

    actions_lib = setuplib("../shuffler/libactions.so")
    action_list2 = get_action_list(actions_lib, [], [], 2)

    action_spaces = [
        ActionSpace(
            name="list2",
            space=Space(
                named_discrete=NamedDiscreteSpace(name=action_list2),
            ),
        ),
    ]

    observation_spaces = [
        ObservationSpace(
            name="runtime_sec",
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
            name="runtime_percent",
            space=Space(
                double_value=DoubleRange(min=0, max=100),
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
            name="base_runtime_sec",
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
            name="base_runtime_percent",
            space=Space(
                double_value=DoubleRange(min=0, max=100),
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
        ObservationSpace(
            name="embedding",
            space=Space(
                byte_sequence=ByteSequenceSpace(length_range=Int64Range(min=0)),
            ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                int64_value=0,
            ),
        ),
    ]

    def __init__(
        self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark
    ):
        super().__init__(working_directory, action_space, benchmark)
        self.parsed_bench = BenchmarkUri.from_string(benchmark.uri)

        self.baseline_size = None
        self.baseline_runtime_sec = None
        self.baseline_runtime_percent = None
        self._lists_valid = True
        self.pass_list = []

        self.bench_name = " ".join(self.parsed_bench.params["bench_name"])
        self.fun_name = " ".join(self.parsed_bench.params["fun_name"])

        self.soc = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM, 0)
        self.instance = 0
        while True:  # Find self instance number and bind to corresponding socket
            try:
                self.soc.bind(f"\0{self.bench_name}:{self.fun_name}_{self.instance}")
            except OSError as e:
                if e.errno != errno.EADDRINUSE:
                    raise
                else:
                    self.instance += 1
                    continue
            break

        self.attach_backend()

        self.get_baseline()

        self.get_state()

        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        action_string = action.string_value
        if action_string == None:
            raise ValueError("Expected pass name, got None")
        logging.info("Applying action %s", action_string)

        if (
            re.match(
                "none_pass",
                action_string[1:] if action_string[0] == ">" else action_string,
            )
            != None
        ):
            return False, None, False

        list_num = get_pass_list(
            self.actions_lib,
            action_string[1:] if action_string[0] == ">" else action_string,
        )
        if list_num != 2:
            raise ValueError(f"Unknown pass {action_string}")

        self.pass_list.append(action_string)

        self._lists_valid = True
        if valid_pass_seq(self.actions_lib, self.pass_list, 2) != 0:
            self._lists_valid = False
            self.pass_list.pop()
            new_space = None
            return True, new_space, True
        else:
            new_list = get_action_list(
                self.actions_lib, [], self.get_list(pass_list), pass_list
            )
            if new_list != []:
                new_space = ActionSpace(
                    name="new_space",
                    space=Space(
                        named_discrete=NamedDiscreteSpace(name=new_list),
                    ),
                )
            else:
                new_space = None

        self.get_state()

        return True if new_space == None else False, new_space, False

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info("Computing observation from space %s", observation_space.name)
        if observation_space.name == "runtime_sec":
            return Event(double_value=self.runtime_sec)
        elif observation_space.name == "runtime_percent":
            return Event(double_value=self.runtime_percent)
        elif observation_space.name == "size":
            return Event(int64_value=self.size)
        elif observation_space.name == "base_runtime_sec":
            return Event(double_value=self.baseline_runtime_sec)
        elif observation_space.name == "base_runtime_percent":
            return Event(double_value=self.baseline_runtime_percent)
        elif observation_space.name == "base_size":
            return Event(int64_value=self.baseline_size)
        elif observation_space.name == "embedding":
            return Event(
                byte_tensor=ByteTensor(shape=[len(self.embedding)], value=self.embedding)
            )
        elif observation_space.name == "passes":
            return Event(
                event_list=ListEvent(
                    event=list(
                        map(lambda name: Event(string_value=name), self.pass_list)
                    )
                )
            )
        else:
            raise KeyError(observation_space.name)

    def get_baseline(self):
        recv_name = self.soc.recv(4096)
        if recv_name.decode("utf-8") != self.fun_name:
            print(
                (
                    f"Got unexpected function name from backend. "
                    f"Expected [{self.fun_name}] got [{recv_name.decode('utf-8')}]"
                ),
                file=sys.stderr,
            )
            raise ValueError("Got unexpected function name from backend")
        self.soc.send(bytes(1))  # Send empty list (plugin will use default passes)
        self.soc.recv(4096)  # Discard embedding message
        rec_data = self.soc.recv(24)
        rec_data = struct.unpack("ddi", rec_data)
        self.baseline_size = rec_data[2]
        self.baseline_runtime_percent = rec_data[0]
        self.baseline_runtime_sec = rec_data[1]

    def get_state(self):
        recv_name = self.soc.recv(4096)
        if recv_name.decode("utf-8") != self.fun_name:
            print(
                (
                    f"Got unexpected function name from backend. "
                    f"Expected [{self.fun_name}] got [{recv_name.decode('utf-8')}]"
                ),
                file=sys.stderr,
            )
            raise ValueError("Got unexpected function name from backend")
        if self.pass_list == []:
            self.soc.send(
                "?".encode("utf-8")
            )  # Send '?' as pass list to get empty list stats
        else:
            list_msg = "\n".join(self.pass_list).encode("utf-8")
            self.soc.send(list_msg)
        self.embedding = self.soc.recv(4096)
        rec_data = self.soc.recv(24)
        rec_data = struct.unpack("ddi", rec_data)
        self.size = rec_data[2]
        self.runtime_percent = rec_data[0]
        self.runtime_sec = rec_data[1]

    def attach_backend(self):
        kernel_dir = f"/tmp/{self.bench_name}:backend_{self.instance}"

        try:
            os.makedirs(kernel_dir)

            if "embedding_length" in self.parsed_bench.params:
                embedding_length = (
                    f"""-e {self.parsed_bench.params["embedding_length"][0]}"""
                )
            else:
                embedding_length = ""

            if "build_string" in self.parsed_bench.params:
                build_string = (
                    f"""-b{" ".join(self.parsed_bench.params["build_string"])}"""
                )
            else:
                build_string = ""

            if "run_string" in self.parsed_bench.params:
                run_string = (
                    f"""-r {" ".join(self.parsed_bench.params["run_string"])}"""
                )
            else:
                run_string = ""

            if "bench_repeats" in self.parsed_bench.params:
                bench_repeats = (
                    f"""--repeats {self.parsed_bench.params["bench_repeats"][0]}"""
                )
            else:
                bench_repeats = ""

            if "plugin_path" in self.parsed_bench.params:
                plugin_path = (
                    f"""-p{"".join(self.parsed_bench.params["plugin_path"])}"""
                )
            else:
                plugin_path = ""

            name_string = f"-n{self.bench_name}"
            instance_num = f"-i{self.instance}"

            # Copy benchmark files to kernel directory
            copytree(self.parsed_bench.path, kernel_dir, dirs_exist_ok=True)

            kernel_bin = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../kernel/gcc-multienv-kernel",
            )
            popen_args = [
                kernel_bin,
                embedding_length,
                build_string,
                run_string,
                plugin_path,
                name_string,
                instance_num,
            ]

            # Start kernel process
            Popen(list(filter(None, popen_args)), cwd=kernel_dir)

            # Wait for it to set up socket and connect to it
            while True:
                try:
                    self.soc.connect(f"\0{self.bench_name}:backend_{self.instance}")
                except ConnectionRefusedError:
                    continue
                return

        except FileExistsError:
            while True:
                try:
                    self.soc.connect(f"\0{self.bench_name}:backend_{self.instance}")
                except ConnectionRefusedError:
                    continue
                return

if __name__ == "__main__":
    create_and_run_compiler_gym_service(GccMultienvCompilationSession)
