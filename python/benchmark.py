#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shlex
import subprocess
import tempfile

import cmdstanpy
import posteriordb

logging.basicConfig(level=logging.WARNING)


def setup_model(*, cmdstan_dir, job_dir, name, model, data):
    """Compile Stan model."""
    cmdstanpy.set_cmdstan_path(cmdstan_dir)

    model = model.replace("<-", "=")

    with tempfile.NamedTemporaryFile(
        "w", prefix=f"{name}_", suffix=".stan", dir=job_dir, delete=False
    ) as f:
        print(model, file=f)
        model_file = f.name

    with tempfile.NamedTemporaryFile(
        "w", prefix=f"{name}_", suffix=".json", dir=job_dir, delete=False
    ) as f:
        json.dump(data, f, indent=2, sort_keys=True)
        data_file = f.name

    cmdstanpy.CmdStanModel(stan_file=model_file)
    return model_file, data_file


def setup_posteriordb_models(*, cmdstan_dir, manifest_info, job_dir=None):
    """Compile posteriordb binaries."""
    if job_dir is None:
        job_dir = tempfile.mkdtemp(prefix="jobs_")

    print(f"Building models in {job_dir}")

    # define POSTERIORDB env variable
    # OR
    # hack: assume that posteriordb is installed from GH clone inplace
    # pip install -e .
    pdb_path = os.environ.get(
        "POSTERIORDB",
        os.path.join(
            os.path.dirname(posteriordb.__file__),
            "..",
            "..",
            "..",
            "posterior_database",
        ),
    )
    pdb = posteriordb.PosteriorDatabase(pdb_path)

    print(f"PosteriorDB in {pdb_path}")

    manifest = {
        **manifest_info,
        "jobs": [],
    }

    N = len(pdb.posterior_names())
    print(f"PosteriorDB N models: {N}")
    for n, name in enumerate(pdb.posterior_names(), 1):
        posterior = pdb.posterior(name)
        try:
            print(f"Building model ({n}/{N}): {name}", flush=True)
            model_file, data_file = setup_model(
                cmdstan_dir=cmdstan_dir,
                job_dir=job_dir,
                name=name,
                model=posterior.model.code("stan"),
                data=posterior.data.values(),
            )
            manifest["jobs"].append({"model_file": model_file, "data_file": data_file})
        except Exception as e:
            print(f"\nmodel {name} failed:\n{e}", flush=True)

    with tempfile.NamedTemporaryFile(
        "w", prefix="manifest_", suffix=".json", dir=job_dir, delete=False
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return job_dir, manifest


def setup_cmdstan(
    *,
    cores,
    cmdstan_branch,
    stan_branch,
    math_branch,
    cmdstan_url,
    stan_url,
    math_url,
    cmdstan_dir=None,
):
    """Clone and build CmdStan. Compile model binaries."""
    if cmdstan_dir is None:
        cmdstan_dir = tempfile.mkdtemp(prefix="cmdstan_")

    print(f"Building cmdstan in {cmdstan_dir}")

    build_cmdstan = os.path.join(
        os.path.dirname(__file__), "..", "R", "build_cmdstan.R"
    )
    cmd = (
        f"Rscript {build_cmdstan}"
        f" --cores={cores}"
        f" --cmdstan_branch={cmdstan_branch}"
        f" --stan_branch={stan_branch}"
        f" --math_branch={math_branch}"
        f" --cmdstan_url={cmdstan_url}"
        f" --stan_url={stan_url}"
        f" --math_url={math_url}"
        f" {cmdstan_dir}"
    )
    print(cmd)
    build_cmd = subprocess.run(
        shlex.split(cmd),
        capture_output=True,
    )

    if build_cmd.returncode == 0:
        print("Cmdstan build successfully", flush=True)
    else:
        print(build_cmd.stdout)
        print(build_cmd.stderr, flush=True)
        raise Exception("Cmdstan failed to build")

    # get the cmdstan subfolder
    cmdstan_dir = os.path.join(cmdstan_dir, os.listdir(cmdstan_dir)[0])

    return cmdstan_dir


def main_setup(
    *,
    cores,
    cmdstan_branch,
    stan_branch,
    math_branch,
    cmdstan_url,
    stan_url,
    math_url,
    cmdstan_dir=None,
    job_dir=None,
):

    cmdstan_dir = setup_cmdstan(
        cores=cores,
        cmdstan_branch=cmdstan_branch,
        stan_branch=stan_branch,
        math_branch=math_branch,
        cmdstan_url=cmdstan_url,
        stan_url=stan_url,
        math_url=math_url,
    )

    manifest_info = {
        "cmdstan_branch": cmdstan_branch,
        "stan_branch": stan_branch,
        "math_branch": math_branch,
        "cmdstan_url": cmdstan_url,
        "stan_url": stan_url,
        "math_url": math_url,
    }

    job_dir, manifest = setup_posteriordb_models(
        cmdstan_dir=cmdstan_dir, manifest_info=manifest_info
    )

    return cmdstan_dir, job_dir, manifest


def sample(model_file, data_file, dir, args=None):
    """Run sample."""
    if args is None:
        args = {}
    model_object = CmdStanModel(
        stan_file=model_file, exe_file=os.path.splitext(model_file)[0]
    )
    fit = model_object.sample(data=data, *args)
    fit.save_csvfiles(dir=dir)
    # return a copy
    return fit.runset.csv_files[:]


def main_sample(manifest, args=None, nrounds=1):
    """Run fits for models."""
    fits = []
    fit_dir = tempfile.mkdtemp(prefix="fit_")
    for i, jobs in enumerate(manifest["jobs"], 1):
        job_fits = []
        fit_dir_i = os.path.join(fit_dir, str(i))
        os.makedirs(fit_dir_i)
        for _ in range(nrounds):
            try:
                fit_paths = sample(**jobs, dir=fit_dir_i, args=args)
                job_fits.extend(fit_paths)
            except Exception as e:
                print("Sampling failed: {jobs['model_file']}", flush=True)
        fits.append(job_fits)
    return fits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use",
    )
    parser.add_argument(
        "--cmdstan_branch", default="develop", help="cmdstan repo branch"
    )
    parser.add_argument("--stan_branch", default="develop", help="stan repo branch")
    parser.add_argument("--math_branch", default="develop", help="math repo branch")
    parser.add_argument(
        "--cmdstan_url",
        default="http://github.com/stan-dev/cmdstan",
        help="cmdstan repo url",
    )
    parser.add_argument(
        "--stan_url", default="http://github.com/stan-dev/stan", help="stan repo url"
    )
    parser.add_argument(
        "--math_url", default="http://github.com/stan-dev/math", help="math repo url"
    )

    # Sample args
    parser.add_argument(
        "--nrounds", default=1, help="number of times sampling is done (sample)"
    )
    parser.add_argument("--chains", default=1, help="number of chains (sample)")
    parser.add_argument(
        "--parallel_chains", default=1, help="number of parallel chains (sample)"
    )
    parser.add_argument(
        "--threads_per_chain", default=1, help="threads per chain (sample)"
    )
    parser.add_argument("--seed", default=None, help="Seed (sample)")
    parser.add_argument(
        "--iter_warmup", default=None, help="Number of warmup samples (sample)"
    )
    parser.add_argument(
        "--iter_sampling", default=None, help="Number of samples (sample)"
    )
    parser.add_argument("--thin", default=None, help="Thin (sample)")
    parser.add_argument("--max_treedepth", default=None, help="Max treedepth (sample)")
    parser.add_argument("--metric", default=None, help="Metric (sample)")
    parser.add_argument("--step_size", default=None, help="Step size (sample)")
    parser.add_argument("--adapt_engaged", default=True, help="Adapt engaged (sample)")
    parser.add_argument("--adapt_delta", default=None, help="Adapt delta (sample)")
    parser.add_argument(
        "--adapt_init_phase", default=None, help="Adapt init phase (sample)"
    )
    parser.add_argument(
        "--adapt_metric_window", default=None, help="Adapt metric window (sample)"
    )
    parser.add_argument(
        "--adapt_step_size", default=None, help="Adapt step size (sample)"
    )
    parser.add_argument("--fixed_param", default=False, help="Fixed param (sample)")

    args = parser.parse_args()

    setup_args_defaults = {
        "cores",
        "cmdstan_branch",
        "stan_branch",
        "math_branch",
        "cmdstan_url",
        "stan_url",
        "math_url",
    }
    setup_args = {
        key: value for key, value in vars(args).items() if key in setup_args_defaults
    }
    sample_args = {
        key: value
        for key, value in vars(args).items()
        if key not in setup_args_defaults
    }
    nrounds = sample_args.pop("nrounds", 1)

    cmdstan_dir, job_dir, manifest = main_setup(**setup_args)

    fits = main_sample(manifest=manifest, args=sample_args, nrounds=nrounds)
