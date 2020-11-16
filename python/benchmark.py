#!/usr/bin/env python3
import posteriordb
import os
import argparse
import tempfile
import subprocess
import shlex
import json
import cmdstanpy


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
    job_dir=None,
):
    """Clone and build CmdStan. Compile model binaries."""
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    if cmdstan_dir:
        cmdstan_dir = tempfile.mkdtemp(prefix="cmdstan_")

    print(f"Building cmdstan in {cmdstan_dir}")

    build_cmd = subprocess.run(
        shlex.split(
            f"Rscript R/build_cmdstan.R --cores={cores} --cmdstan_branch={cmdstan_branch} --stan_branch={stan_branch} --math_branch={math_branch} --cmdstan_url={cmdstan_url} --stan_url={stan_url} --math_url={math_url} {cmdstan_dir}"
        )
    )

    if build_cmd.returncode == 0:
        print("Cmdstan build successfully")
    else:
        raise Exception("Cmdstan failed to build")

    cmdstanpy.set_cmdstan_path(cmdstan_dir)

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
            os.path.basename(posteriordb.__file__),
            "..",
            "..",
            "..",
            "posteriordb",
            "posterior_database",
        ),
    )
    pdb = posteriordb.PosteriorDatabase(pdb_path)

    manifest = {
        "cmdstan_branch": cmdstan_branch,
        "stan_branch": stan_branch,
        "math_branch": math_branch,
        "cmdstan_url": cmdstan_url,
        "stan_url": stan_url,
        "math_url": math_url,
        "jobs": [],
    }

    for name in pdb.posterior_names():
        posterior = pdb.posterior(name)

        with tempfile.NamedTemporaryFile(
            "w", prefix=f"{name}_", suffix=".stan", dir=job_dir, delete=False
        ) as f:
            print(posterior.model.code("stan"), file=f)
            model_file = f.name

        with tempfile.NamedTemporaryFile(
            "w", prefix=f"{name}_", suffix=".json", dir=job_dir, delete=False
        ) as f:
            json.dump(posterior.data.values(), f, indent=2, sort_keys=True)
            data_file = f.name

        try:
            cmdstanpy.CmdStanModel(stan_file=model_file)
        except Exception as e:
            print(f"model {model_file}, data {data_file} failed:\n{e}")
            continue

        manifest["jobs"].append({model_file: model_file, data_file: data_file})

    with tempfile.NamedTemporaryFile(
        "w", prefix="manifest_", suffix=".json", dir=job_dir, delete=False
    ) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return cmdstan_dir, job_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use (default: %(default)s)",
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

    args = parser.parse_args()

    # setup_cmdstan(**vars(args))
