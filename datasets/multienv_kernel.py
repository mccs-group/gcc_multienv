from compiler_gym.datasets import Benchmark, Dataset, BenchmarkUri
from typing import Iterable


class MultienvDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://multienv_kernel",
            license="N/A",
            description=(
                "Just feed anything compilable and runnable into here, "
                "and the environment will figure everything out for you*\n\n"
                "*not guaranteed"
            ),
            validatable="No",
        )

    def benchmark_uris(self) -> Iterable[str]:
        return ["Hippety Hopper (not really a benchmark)"]

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        return Benchmark.from_file_contents(uri, None)
