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

def sample(dir, model_file, data_file, exe_file, args=None):
    """Run sample."""
    if args is None:
        args = {}
    model_object = cmdstanpy.CmdStanModel(
        stan_file=model_file,
        exe_file=exe_file,
    )
    fit = model_object.sample(data=data_file, **args)
    fit.save_csvfiles(dir = dir)

    return fit.runset.csv_files

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("configuration", help="json configuration file for experiment")
    parser.add_argument("build_dir", default=None, help="build directory")
    parser.add_argument("--run_dir", default=None, help="run directory")
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use",
    )

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
    if not os.path.exists(args.build_dir):
        raise Exception(f"Build directory does not exist: {build_dir}")

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = tempfile.mkdtemp(prefix="run_")
    os.makedirs(run_dir, exist_ok=True)

    with open(args.configuration, "r") as f:
        configurations = json.load(f)

    posteriors = configurations["posteriors"]

    if len(posteriors) == 0:
        posteriors = pdb.posterior_names()

    # Locate the built cmdstans and models
    jobs = []
    for cmdstan in configurations["cmdstans"]:
        cmdstan_commit = get_head_commit(cmdstan["cmdstan_url"], cmdstan["cmdstan_branch"])
        stan_commit = get_head_commit(cmdstan["stan_url"], cmdstan["stan_branch"])
        math_commit = get_head_commit(cmdstan["math_url"], cmdstan["math_branch"])

        cmdstan_dir = find_cmdstan(
            dir = build_dir,
            cmdstan_commit=cmdstan_commit,
            stan_commit=stan_commit,
            math_commit=math_commit,
            cmdstan_url=cmdstan["cmdstan_url"],
            stan_url=cmdstan["stan_url"],
            math_url=cmdstan["math_url"]
        )

        if cmdstan_dir == None:
            raise Exception(f"Necessary cmdstan not found. Check {build_dir} exists, and possibly re-run build script")

        models = {}
        for posterior in posteriors:
            model = find_model(build_dir, cmdstan_dir, posterior)

            if model == None:
                raise Exception(f"Model missing. Check {build_dir} exists, and possibly re-run build script")

            models[posterior] = model

            jobs.append({
                "cmdstan_dir" : cmdstan_dir,
                "cmdstan_info" : {
                    "cmdstan_commit" : cmdstan_commit,
                    "stan_commit" : stan_commit,
                    "math_commit" : math_commit,
                    "cmdstan_url" : cmdstan["cmdstan_url"],
                    "stan_url" : cmdstan["stan_url"],
                    "math_url" : cmdstan["math_url"]
                },
                **model
            })

    fits = {}
    job_dir = tempfile.mkdtemp(prefix="job_", dir = run_dir)
    manifest = {"jobs": []}
    order = args.nrounds * list(range(len(jobs)))
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
            sample_args["chain_ids"] = count[i] + 1
            chains = sample_args.get("chains", 1)
            count[i] += chains
            fit = sample(
                    fit_dir, job["model_file"], job["data_file"], job["exe_file"], args=sample_args
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
                        "args": sample_args,
                        "cmdstan_info": job["cmdstan_info"],
                    }
                )
        except Exception as e:
            print(
                f'Sampling failed: {job["name"]} -> cmdstan {job["cmdstan_dir"]}:\n{e}',
                flush=True,
            )

    with tempfile.NamedTemporaryFile(
        "w", prefix="manifest_", suffix=".json", dir=run_dir, delete=False
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        manifest_path = f.name
