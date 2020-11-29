#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
import random
import shlex
import subprocess
import tempfile
from collections import Counter

import cmdstanpy
import posteriordb

logging.basicConfig(level=logging.WARNING)

pdb = posteriordb.PosteriorDatabaseGithub(overwrite=True)

print(f"PosteriorDB in {str(pdb.path)}", flush=True)

def setup_model(*, cmdstan_dir, model_dir, name, model, data):
    """Compile Stan model."""
    cmdstanpy.set_cmdstan_path(cmdstan_dir)

    model = model.replace("<-", "=")

    with tempfile.NamedTemporaryFile(
        "w", prefix=f"{name}_", suffix=".stan", dir=model_dir, delete=False
    ) as f:
        print(model, file=f)
        model_file = f.name

    with tempfile.NamedTemporaryFile(
        "w", prefix=f"{name}_", suffix=".json", dir=model_dir, delete=False
    ) as f:
        json.dump(data, f, indent=2, sort_keys=True)
        data_file = f.name

    model_object = cmdstanpy.CmdStanModel(stan_file=model_file)
    exe_file = model_object.exe_file
    return model_file, data_file, exe_file


def find_model(dir, cmdstan_dir, model):
    benchmark_manifests = glob.glob(os.path.join(dir, "manifest_*.json"))
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


def setup_posteriordb_models(*, posteriors, dir, cmdstan_dir, model_dir=None):
    """Compile posteriordb binaries."""
    models = []

    new_posteriors = []
    for posterior in posteriors:
        model = find_model(dir, cmdstan_dir, posterior)
        if not model:
            new_posteriors.append(posterior)
        else:
            models.append(model)

    if len(new_posteriors) == 0:
        return None, models

    if model_dir is None:
        model_dir = tempfile.mkdtemp(prefix="model_", dir=dir)

    print(f"Building models in {model_dir}")

    manifest = {
        "models": {},
    }

    N = len(new_posteriors)
    print(f"PosteriorDB N models: {N}")
    for n, name in enumerate(new_posteriors, 1):
        try:
            posterior = pdb.posterior(name)
            print(f"Building model ({n}/{N}): {name}", flush=True)
            model_file, data_file, exe_file = setup_model(
                cmdstan_dir=cmdstan_dir,
                model_dir=model_dir,
                name=name,
                model=posterior.model.code("stan"),
                data=posterior.data.values(),
            )
            model = {
                "name": name,
                "model_file": model_file,
                "data_file": data_file,
                "exe_file": exe_file,
            }
            manifest["models"][name] = model
            models.append(model)
        except Exception as e:
            print(f"\nmodel {name} failed:\n{e}", flush=True)

    with tempfile.NamedTemporaryFile(
        "w", prefix="manifest_", suffix=".json", dir=model_dir, delete=False
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return model_dir, models

def get_head_commit(url, branch):
    cmd = f"git ls-remote --heads {url} {branch}"
    run = subprocess.run(
        shlex.split(cmd),
        capture_output=True,
    )

    if run.returncode != 0:
        print(f"stdout: {run.stdout}")
        print(f"stderr: {run.stderr}", flush=True)
        raise Exception(f"Exception running: {cmd}")

    out = run.stdout.decode('utf-8').strip().split()

    if len(out) == 0:
        print(f"stdout: {run.stdout}")
        print(f"stderr: {run.stderr}", flush=True)
        raise Exception(f"Expected git commit id, found nothing in: {cmd}")

    return out[0]

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


def setup_cmdstan(
    *,
    dir,
    cores,
    cmdstan_branch,
    stan_branch,
    math_branch,
    cmdstan_url,
    stan_url,
    math_url,
    cmdstan_dir=None,
):
    cmdstan_commit = get_head_commit(cmdstan_url, cmdstan_branch)
    stan_commit = get_head_commit(stan_url, stan_branch)
    math_commit = get_head_commit(math_url, math_branch)

    """Clone and build CmdStan. Compile model binaries."""
    # Search for pre-existing cmdstan
    cmdstan_dir_found = find_cmdstan(
        dir, cmdstan_commit, stan_commit, math_commit, cmdstan_url, stan_url, math_url
    )

    if cmdstan_dir_found:
        return cmdstan_dir_found

    if cmdstan_dir is None:
        cmdstan_dir = tempfile.mkdtemp(prefix="cmdstan_", dir=dir)

    print(f"Building cmdstan in {cmdstan_dir}")

    build_cmdstan = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "R", "build_cmdstan.R")
    )

    cmdstan_clone = f"git clone --depth=1 --single-branch --branch={cmdstan_branch} {cmdstan_url} {cmdstan_dir}"
    stan_clone = f"git clone --depth=1 --single-branch --branch={stan_branch} {stan_url} {cmdstan_dir}/stan"
    math_clone = f"git clone --depth=1 --single-branch --branch={math_branch} {math_url} {cmdstan_dir}/stan/lib/stan_math"

    print(cmdstan_clone)
    cmdstan_clone_cmd = subprocess.run(
        shlex.split(cmdstan_clone),
        capture_output=True,
    )

    if cmdstan_clone_cmd.returncode == 0:
        print("Cmdstan clone successful", flush=True)
    else:
        print(cmdstan_clone_cmd.stdout)
        print(cmdstan_clone_cmd.stderr, flush=True)
        raise Exception("Cmdstan failed to clone")

    print(stan_clone)
    stan_clone_cmd = subprocess.run(
        shlex.split(stan_clone),
        capture_output=True,
    )

    if stan_clone_cmd.returncode == 0:
        print("Stan clone successful", flush=True)
    else:
        print(stan_clone_cmd.stdout)
        print(stan_clone_cmd.stderr, flush=True)
        raise Exception("Stan failed to clone")

    print(math_clone)
    math_clone_cmd = subprocess.run(
        shlex.split(math_clone),
        capture_output=True,
    )

    if math_clone_cmd.returncode == 0:
        print("Math clone successful", flush=True)
    else:
        print(math_clone_cmd.stdout)
        print(math_clone_cmd.stderr, flush=True)
        raise Exception("Math failed to clone")

    build = f"Rscript {build_cmdstan}" f" --cores={cores}" f" {cmdstan_dir}"

    print(build)
    build_cmd = subprocess.run(
        shlex.split(build),
        capture_output=True,
    )

    if build_cmd.returncode == 0:
        print("Cmdstan build successfully", flush=True)
    else:
        print(build_cmd.stdout)
        print(build_cmd.stderr, flush=True)
        raise Exception("Cmdstan failed to build")

    manifest = {
        "cmdstan_commit": cmdstan_commit,
        "stan_commit": stan_commit,
        "math_commit": math_commit,
        "cmdstan_branch": cmdstan_branch,
        "stan_branch": stan_branch,
        "math_branch": math_branch,
        "cmdstan_url": cmdstan_url,
        "stan_url": stan_url,
        "math_url": math_url,
    }

    with tempfile.NamedTemporaryFile(
        "w", prefix="manifest_", suffix=".json", dir=cmdstan_dir, delete=False
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return cmdstan_dir


def main_setup(
    *,
    dir,
    posteriors,
    cores,
    cmdstan_branch,
    stan_branch,
    math_branch,
    cmdstan_url,
    stan_url,
    math_url,
    cmdstan_dir=None,
    model_dir=None,
):
    cmdstan_info = dict(
        dir=dir,
        cores=cores,
        cmdstan_branch=cmdstan_branch,
        stan_branch=stan_branch,
        math_branch=math_branch,
        cmdstan_url=cmdstan_url,
        stan_url=stan_url,
        math_url=math_url,
    )
    cmdstan_dir = setup_cmdstan(**cmdstan_info)

    model_dir, models = setup_posteriordb_models(
        posteriors=posteriors, dir=dir, cmdstan_dir=cmdstan_dir
    )

    if model_dir:
        manifest = {"cmdstan_dir": cmdstan_dir, "model_dir": model_dir}

        with tempfile.NamedTemporaryFile(
            "w", prefix="manifest_", suffix=".json", dir=dir, delete=False
        ) as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    return [
        {"cmdstan_dir": cmdstan_dir, **model, "cmdstan_info": cmdstan_info}
        for model in models
    ]


def sample(dir, model_file, data_file, exe_file, args=None):
    """Run sample."""
    if args is None:
        args = {}
    model_object = cmdstanpy.CmdStanModel(
        stan_file=model_file,
        exe_file=exe_file,
    )
    fit = model_object.sample(data=data_file, **args)
    fit.save_csvfiles(dir=dir)

    return fit.runset.csv_files


def main_sample(dir, jobs, args=None, nrounds=1):
    """Run fits for models."""
    fits = {}
    job_dir = tempfile.mkdtemp(prefix="job_", dir=dir)
    manifest = {"jobs": []}
    order = nrounds * list(range(len(jobs)))
    random.shuffle(order)
    count = Counter()
    fit_dirs = {}
    for i in order:
        job = jobs[i]
        if i not in fit_dirs:
            fit_dir = tempfile.mkdtemp(prefix="fit_", dir=job_dir)
            fit_dirs[i] = fit_dir
        else:
            fit_dir = fit_dirs[i]
        try:
            # offset for unique filename
            args["chain_ids"] = count[i] + 1
            chains = args.get("chains", 1)
            count[i] += chains
            fit = sample(
                fit_dir, job["model_file"], job["data_file"], job["exe_file"], args=args
            )
            if job["name"] not in fits:
                fits[job["name"]] = []
            fits[job["name"]].append(fit)
            manifest["jobs"].append(
                {
                    "cmdstan_dir": job["cmdstan_dir"],
                    "name": job["name"],
                    "fit_files": fit,
                    "fit_dir": fit_dir,
                    "args": args,
                    "cmdstan_info": job["cmdstan_info"],
                }
            )
        except Exception as e:
            print(
                f'Sampling failed: {job["name"]} -> cmdstan {job["cmdstan_dir"]}:\n{e}',
                flush=True,
            )

    with tempfile.NamedTemporaryFile(
        "w", prefix="manifest_", suffix=".json", dir=dir, delete=False
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        manifest_path = f.name

    return fits, manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("configuration", help="json configuration file for experiment")
    parser.add_argument("--build_dir", default=None, help="build directory")
    parser.add_argument("--run_dir", default=None, help="run directory")
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use",
    )

    # Sample args
    parser.add_argument(
        "--nrounds",
        default=1,
        type=int,
        help="number of times sampling is done (sample)",
    )
    parser.add_argument(
        "--chains", default=1, type=int, help="number of chains (sample)"
    )
    parser.add_argument(
        "--parallel_chains",
        default=1,
        type=int,
        help="number of parallel chains (sample)",
    )
    parser.add_argument(
        "--threads_per_chain", default=1, type=int, help="threads per chain (sample)"
    )
    parser.add_argument("--seed", default=None, type=int, help="Seed (sample)")
    parser.add_argument(
        "--iter_warmup",
        default=None,
        type=int,
        help="Number of warmup samples (sample)",
    )
    parser.add_argument(
        "--iter_sampling", default=None, type=int, help="Number of samples (sample)"
    )
    parser.add_argument("--thin", default=None, type=int, help="Thin (sample)")
    parser.add_argument(
        "--max_treedepth", default=None, type=int, help="Max treedepth (sample)"
    )
    parser.add_argument("--metric", default=None, help="Metric (sample)")
    parser.add_argument(
        "--step_size", default=None, type=float, help="Step size (sample)"
    )
    parser.add_argument(
        "--adapt_engaged", default=True, type=bool, help="Adapt engaged (sample)"
    )
    parser.add_argument(
        "--adapt_delta", default=None, type=float, help="Adapt delta (sample)"
    )
    parser.add_argument(
        "--adapt_init_phase", default=None, type=int, help="Adapt init phase (sample)"
    )
    parser.add_argument(
        "--adapt_metric_window",
        default=None,
        type=int,
        help="Adapt metric window (sample)",
    )
    parser.add_argument(
        "--adapt_step_size", default=None, type=int, help="Adapt step size (sample)"
    )
    parser.add_argument(
        "--fixed_param", default=False, type=bool, help="Fixed param (sample)"
    )

    args = parser.parse_args()

    setup_args_defaults = {"build_dir", "run_dir", "configuration", "cores"}
    setup_args = {
        key: value for key, value in vars(args).items() if key in setup_args_defaults
    }

    sample_args_defaults = {
        "chains",
        "parallel_chains",
        "threads_per_chain",
        "seed",
        "iter_warmup",
        "iter_sampling",
        "thin",
        "max_treedepth",
        "metric",
        "step_size",
        "adapt_engaged",
        "adapt_delta",
        "adapt_init_phase",
        "adapt_metric_window",
        "adapt_step_size",
        "fixed_param",
    }

    sample_args = {
        key: value for key, value in vars(args).items() if key in sample_args_defaults
    }

    build_dir = args.build_dir
    if build_dir is None:
        build_dir = tempfile.mkdtemp(prefix="build_")
    os.makedirs(build_dir, exist_ok=True)

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="run_")
    os.makedirs(run_dir, exist_ok=True)

    with open(args.configuration, "r") as f:
        configurations = json.load(f)

    posteriors = configurations["posteriors"]

    if len(posteriors) == 0:
        posteriors = pdb.posterior_names()

    jobs = []
    for cmdstan in configurations["cmdstans"]:
        jobs.extend(
            main_setup(
                dir=build_dir,
                posteriors=posteriors,
                cores=args.cores,
                cmdstan_branch=cmdstan["cmdstan_branch"],
                stan_branch=cmdstan["stan_branch"],
                math_branch=cmdstan["math_branch"],
                cmdstan_url=cmdstan["cmdstan_url"],
                stan_url=cmdstan["stan_url"],
                math_url=cmdstan["math_url"],
            )
        )

    fits, manifest = main_sample(
        dir=args.run_dir, jobs=jobs, args=sample_args, nrounds=args.nrounds
    )
    print(fits)
    print(manifest)
