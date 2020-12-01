import glob
import json
import os

def find_model(build_dir, cmdstan_dir, model):
    benchmark_manifests = glob.glob(os.path.join(build_dir, "manifest_*.json"))
    for benchmark_manifest_file in benchmark_manifests:
        with open(benchmark_manifest_file, "r") as f:
            benchmark_manifest = json.load(f)

        if benchmark_manifest["cmdstan_dir"] == cmdstan_dir:
            model_manifests = glob.glob(
                os.path.join(benchmark_manifest["model_dir"], "manifest_*.json")
            )
            for model_manifest_file in model_manifests:
                with open(model_manifest_file, "r") as f:
                    model_manifest = json.load(f)

                if model in model_manifest["models"]:
                    print(f"Found model '{model}' for cmdstan '{cmdstan_dir}'")
                    return model_manifest["models"][model]

    return None