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

from find_model import find_model
from find_cmdstan import find_cmdstan
from get_head_commit import get_head_commit

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

def setup_posteriordb_models(*, posteriors, pdb, build_dir, cmdstan_dir):
    """Compile posteriordb binaries."""
    models = {}

    new_posteriors = []
    for posterior in posteriors:
        model = find_model(build_dir, cmdstan_dir, posterior)
        if not model:
            new_posteriors.append(posterior)
        else:
            models[posterior] = model

    if len(new_posteriors) == 0:
        return None

    model_dir = tempfile.mkdtemp(prefix="model_", dir=build_dir)

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
            models[name] = model
        except Exception as e:
            print(f"\nmodel {name} failed:\n{e}", flush=True)

    with tempfile.NamedTemporaryFile(
        "w", prefix="manifest_", suffix=".json", dir=model_dir, delete=False
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return model_dir

def setup_cmdstan(
    *,
    build_dir,
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
        build_dir, cmdstan_commit, stan_commit, math_commit, cmdstan_url, stan_url, math_url
    )

    if cmdstan_dir_found:
        return cmdstan_dir_found

    if cmdstan_dir is None:
        cmdstan_dir = tempfile.mkdtemp(prefix="cmdstan_", dir=build_dir)

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

    build = f"Rscript {build_cmdstan} --cores={cores} {cmdstan_dir}"

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("configuration", help="json configuration file for experiment")
    parser.add_argument("--build_dir", default=None, help="build directory")
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use",
    )

    args = parser.parse_args()

    # If no build dir specified, make one
    build_dir = args.build_dir
    if build_dir is None:
        build_dir = tempfile.mkdtemp(prefix="build_")
    os.makedirs(build_dir, exist_ok=True)

    with open(args.configuration, "r") as f:
        configurations = json.load(f)

    posteriors = configurations["posteriors"]

    # If no posteriors specified, use them all
    if len(posteriors) == 0:
        posteriors = pdb.posterior_names()

    for cmdstan in configurations["cmdstans"]:
        cmdstan_dir = setup_cmdstan(
            build_dir = build_dir,
            cores = args.cores,
            cmdstan_branch=cmdstan["cmdstan_branch"],
            stan_branch=cmdstan["stan_branch"],
            math_branch=cmdstan["math_branch"],
            cmdstan_url=cmdstan["cmdstan_url"],
            stan_url=cmdstan["stan_url"],
            math_url=cmdstan["math_url"]
        )

        model_dir = setup_posteriordb_models(
            posteriors=posteriors, pdb=pdb, build_dir=build_dir, cmdstan_dir=cmdstan_dir
        )

        if model_dir:
            manifest = {"cmdstan_dir" : cmdstan_dir, "model_dir": model_dir}

            with tempfile.NamedTemporaryFile(
                "w", prefix="manifest_", suffix=".json", dir=build_dir, delete=False
            ) as f:
                json.dump(manifest, f, indent=2, sort_keys=True)