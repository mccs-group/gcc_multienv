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
    Int64SequenceSpace,
    Int64Tensor,
    DoubleSequenceSpace,
    DoubleTensor,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from shutil import copytree, copy2, rmtree
from compiler_gym.datasets import BenchmarkUri, Benchmark
from subprocess import *
from time import *
from compiler_gym.envs.gcc_multienv.shuffler import *
from compiler_gym.envs.gcc_multienv.embedding import *
import os, sys
import re
import socket
import errno
import struct
import hashlib
import base64


class GccMultienvCompilationSession(CompilationSession):
    compiler_version: str = "7.3.0"

    actions_lib = setuplib("../shuffler/libactions.so")
    action_list2 = get_list_by_list_num(actions_lib, 2)

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
                double_sequence=DoubleSequenceSpace(
                    length_range=Int64Range(min=0, max=149)
                ),
            ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                double_tensor=DoubleTensor(shape=[1], value=[0.0])
            ),
        ),
        ObservationSpace(
            name="base_embedding",
            space=Space(
                double_sequence=DoubleSequenceSpace(length_range=Int64Range(min=0)),
            ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                double_value=0.0,
            ),
        ),
    ]

    def __init__(
        self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark
    ):
        super().__init__(working_directory, action_space, benchmark)
        self.parsed_bench = BenchmarkUri.from_string(benchmark.uri)

        self.EMBED_LEN_MULTIPLIER = 200

        self.baseline_size = None
        self.baseline_runtime_sec = None
        self.baseline_runtime_percent = None
        self.baseline_embedding = None
        self._lists_valid = True
        self.pass_list = []
        self.indented_pass_list = []
        self.current_action_space = self.action_spaces[0]
        self.embedding = None
        self.orig_properties = None
        self.custom_properties = None

        self.bench_name = " ".join(self.parsed_bench.params["bench_name"])
        self.fun_name = " ".join(self.parsed_bench.params["fun_name"])

        self.soc = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM, 0)
        self.instance = 0
        avail_length = 107 - len(self.bench_name) - len(str(self.instance)) - 2
        if len(self.fun_name) > avail_length:
            name_hash = hashlib.sha256(self.fun_name.encode("utf-8")).digest()
            self.sock_fun_name = base64.b64encode(
                name_hash, "-_".encode("utf-8")
            ).decode("utf-8")
        else:
            self.sock_fun_name = self.fun_name
        while True:  # Find self instance number and bind to corresponding socket
            try:
                self.soc.bind(
                    f"\0{self.bench_name}:{self.sock_fun_name}_{self.instance}"
                )
            except OSError as e:
                if e.errno != errno.EADDRINUSE:
                    raise
                else:
                    self.instance += 1
                    continue
            break

        self.attach_backend()

        self.orig_properties, self.custom_properties = get_property_by_history(
            self.actions_lib, self.pass_list, 2
        )

        self.get_baseline()

        self.get_state()

        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        if action.string_value != "":
            action_string = action.string_value
        else:
            action_string = self.action_spaces[0].space.named_discrete.name[
                action.int64_value
            ]

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

        self.orig_properties, self.custom_properties = get_property_by_history(
            self.actions_lib, self.pass_list, 2
        )

        self.pass_list.append(action_string)

        self._lists_valid = True
        if valid_pass_seq(self.actions_lib, self.pass_list, 2) != 0:
            self._lists_valid = False
            self.pass_list.pop()
            new_space = None
            return True, new_space, True

        if action_string == "fix_loops":
            self.indented_pass_list.append("fix_loops")
            self.indented_pass_list.append("loop")
            self.indented_pass_list.append(">loopinit")
        elif in_loop(self.actions_lib, self.custom_properties):
            self.indented_pass_list.append(">" + action_string)
        else:
            self.indented_pass_list.append(action_string)

        self.get_state()

        return False, None, False

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
                double_tensor=DoubleTensor(
                    shape=[len(self.embedding)], value=self.embedding
                )
            )
        elif observation_space.name == "base_embedding":
            return Event(
                double_tensor=DoubleTensor(
                    shape=[len(self.baseline_embedding)], value=self.baseline_embedding
                )
            )
        elif observation_space.name == "passes":
            return Event(
                event_list=ListEvent(
                    event=list(
                        map(
                            lambda name: Event(string_value=name),
                            self.indented_pass_list,
                        )
                    )
                )
            )
        else:
            raise KeyError(observation_space.name)

    def padded_recv(self, size):
        while True:
            data = self.soc.recv(size)
            if data != bytes(0):
                return data

    def get_baseline(self):
        logging.debug("Getting baseline")
        self.soc.send(bytes(1))  # Send empty list (plugin will use default passes)
        logging.debug("Sent first list")
        embedding_msg = self.padded_recv(1024 * self.EMBED_LEN_MULTIPLIER)
        logging.debug("Got embedding")
        embedding = [x[0] for x in struct.iter_unpack("i", embedding_msg)]
        self.baseline_embedding = self.calc_embedding(embedding) + [
            self.orig_properties,
            self.custom_properties,
        ]
        rec_data = self.padded_recv(24)
        logging.debug("Got profiling data")
        rec_data = struct.unpack("ddi", rec_data)
        self.baseline_size = rec_data[2]
        self.baseline_runtime_percent = rec_data[0]
        self.baseline_runtime_sec = rec_data[1]
        logging.debug("Got all baseline")

    def get_state(self):
        logging.debug("Getting state")
        if self.indented_pass_list == []:
            self.soc.send(
                "?".encode("utf-8")
            )  # Send '?' as pass list to get empty list stats
        else:
            list_msg = ("\n".join(self.indented_pass_list) + "\n").encode("utf-8")
            self.soc.send(list_msg)
        embedding_msg = self.padded_recv(1024 * self.EMBED_LEN_MULTIPLIER)
        logging.debug("Got embedding")
        embedding = [x[0] for x in struct.iter_unpack("i", embedding_msg)]
        self.embedding = self.calc_embedding(embedding) + [
            self.orig_properties,
            self.custom_properties,
        ]
        rec_data = self.padded_recv(24)
        logging.debug("Got profiling data")
        rec_data = struct.unpack("ddi", rec_data)
        self.size = rec_data[2]
        self.runtime_percent = rec_data[0]
        self.runtime_sec = rec_data[1]
        logging.debug("Got all state")

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

            build_string = build_string.replace("lstdc", "lstdc++")

            run_arr = []
            if "run_string" in self.parsed_bench.params:
                for rstr in self.parsed_bench.params["run_string"]:
                    run_string = f"""-r{rstr}"""
                    run_arr.append(run_string)

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
                *run_arr,
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

    def calc_embedding(self, embedding):
        autophase = embedding[:47]
        cfg_len = embedding[47]
        cfg = embedding[48 : 48 + cfg_len]
        val_flow = embedding[48 + cfg_len :]

        cfg_embedding = list(get_flow2vec_embed(cfg, 25))
        val_flow_embedding = list(get_flow2vec_embed(val_flow, 25))

        return autophase + cfg_embedding + val_flow_embedding


if __name__ == "__main__":
    create_and_run_compiler_gym_service(GccMultienvCompilationSession)
