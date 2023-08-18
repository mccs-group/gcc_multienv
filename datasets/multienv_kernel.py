from compiler_gym.datasets import Benchmark, Dataset, BenchmarkUri
from typing import Iterable
from pathlib import Path
import os


class MultienvDataset(Dataset):
    def __init__(self):
        super().__init__(
            name="benchmark://multienv",
            license="N/A",
            description="Whatever",
            validatable="No",
        )
        self._path = None
        self.benches = []
        self._plugin = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = Path(value)

    @property
    def plugin(self):
        return self._plugin

    @plugin.setter
    def plugin(self, value):
        self._plugin = Path(value)

    def parse_benchmarks(self):
        if self._path == None:
            self.benches == ['']
            return
        else:
            self.benches = []
            datafiles = self._path.glob("*/benchmark_info.txt")
            for file in datafiles:
                self.parse_file(file)

    def parse_file(self, file: Path):
        uri_dataset = "multienv_kernel"
        uri_path = str(file.resolve().parent) + "/"
        lines = [x.strip() for x in file.read_text().splitlines()]
        if "build:" in lines:
            uri_build = "build_string=" + lines[lines.index("build:") + 1] + "&"
        else:
            uri_build = ""

        if "run:" in lines:
            uri_run = "run_string=" + lines[lines.index("run:") + 1] + "&"
        else:
            uri_run = ""

        if "embedding_length:" in lines:
            uri_embedding_length = (
                "embedding_length=" + lines[lines.index("embedding_length:") + 1] + "&"
            )
        else:
            uri_embedding_length = ""

        if "bench_repeats:" in lines:
            uri_bench_repeats = (
                "bench_repeats=" + lines[lines.index("bench_repeats:") + 1] + "&"
            )
        else:
            uri_bench_repeats = ""

        if self._plugin != None:
            uri_plugin = "plugin_path=" + str(self._plugin) + "&"
        else:
            uri_plugin = ""

        uri_bench_name = "bench_name=" + str(file.parts[-2]) + "&"

        if "functions:" in lines:
            index = lines.index("functions:") + 1
            for line in lines[index:]:
                bench = (
                    uri_dataset
                    + uri_path
                    + "?"
                    + uri_build
                    + uri_run
                    + uri_embedding_length
                    + uri_bench_repeats
                    + uri_plugin
                    + uri_bench_name
                )
                bench += "fun_name=" + line
                self.benches.append(bench)
        else:
            print(f"No function data found for bench {file.parent}")

    def benchmark_uris(self) -> Iterable[str]:
        if self.benches == []:
            self.parse_benchmarks()
        return self.benches

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        return Benchmark.from_file_contents(uri, None)
