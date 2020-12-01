import glob
import json
import os

def find_cmdstan(
    dir, cmdstan_commit, stan_commit, math_commit, cmdstan_url, stan_url, math_url
):
    benchmark_manifests = glob.glob(os.path.join(dir, "manifest_*.json"))
    for benchmark_manifest_file in benchmark_manifests:
        with open(benchmark_manifest_file, "r") as f:
            benchmark_manifest = json.load(f)

        cmdstan_manifests = glob.glob(
            os.path.join(benchmark_manifest["cmdstan_dir"], "manifest_*.json")
        )
        for cmdstan_manifest_file in cmdstan_manifests:
            with open(cmdstan_manifest_file, "r") as f:
                cmdstan_manifest = json.load(f)

            if (
                { "cmdstan_commit", "stan_commit", "math_commit", "cmdstan_url", "stan_url", "math_url" }.issubset(cmdstan_manifest.keys())
                and cmdstan_manifest["cmdstan_commit"] == cmdstan_commit
                and cmdstan_manifest["stan_commit"] == stan_commit
                and cmdstan_manifest["math_commit"] == math_commit
                and cmdstan_manifest["cmdstan_url"] == cmdstan_url
                and cmdstan_manifest["stan_url"] == stan_url
                and cmdstan_manifest["math_url"] == math_url
            ):
                return benchmark_manifest["cmdstan_dir"]

    return None