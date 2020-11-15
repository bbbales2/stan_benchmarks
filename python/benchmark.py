#!/usr/bin/env python3
import posteriordb
import os
import argparse
import tempfile
import subprocess
import shlex
import json
import cmdstanpy

parser = argparse.ArgumentParser()

parser.add_argument("--cores", type = int, default = 1, help = "Number of cores to use")
parser.add_argument("--cmdstan_branch", default = "develop", help = "cmdstan repo branch")
parser.add_argument("--stan_branch", default = "develop", help = "stan repo branch")
parser.add_argument("--math_branch", default = "develop", help = "math repo branch")
parser.add_argument("--cmdstan_url", default = "http://github.com/stan-dev/cmdstan", help = "cmdstan repo url")
parser.add_argument("--stan_url", default = "http://github.com/stan-dev/stan", help = "stan repo url")
parser.add_argument("--math_url", default = "http://github.com/stan-dev/math", help = "math repo url")

args = parser.parse_args()

# Clone and build cmdstan
if not os.path.exists("tmp"):
    os.mkdir("tmp")

cmdstan_dir = tempfile.mkdtemp(dir = "tmp")

print("Building cmdstan in ", cmdstan_dir)

build_cmd = subprocess.run(shlex.split("Rscript R/build_cmdstan.R --cores={cores} --cmdstan_branch={cmdstan_branch} --stan_branch={stan_branch} --math_branch={math_branch} --cmdstan_url={cmdstan_url} --stan_url={stan_url} --math_url={math_url} {dir}"
                                       .format(dir = cmdstan_dir,
                                               cores = args.cores,
                                               cmdstan_branch = args.cmdstan_branch,
                                               stan_branch = args.stan_branch,
                                               math_branch = args.math_branch,
                                               cmdstan_url = args.cmdstan_url,
                                               stan_url = args.stan_url,
                                               math_url = args.math_url)))

if build_cmd.returncode == 0:
    print("Cmdstan build successfully")
else:
    raise Exception("Cmdstan failed to build")

cmdstanpy.set_cmdstan_path(cmdstan_dir)

# Build job directory with model binaries
if not os.path.exists("jobs"):
    os.mkdir("jobs")

job_dir = tempfile.mkdtemp(dir = "jobs")

pdb_path = os.path.join(os.getcwd(), "posteriordb", "posterior_database")
pdb = posteriordb.PosteriorDatabase(pdb_path)

print("Building models in ", job_dir)

manifest = {
    "cmdstan_branch" : args.cmdstan_branch,
    "stan_branch" : args.stan_branch,
    "math_branch" : args.math_branch,
    "cmdstan_url" : args.cmdstan_url,
    "stan_url" : args.stan_url,
    "math_url" : args.math_url,
    "jobs" : []
}

for name in pdb.posterior_names():
    post = pdb.posterior(name)

    f = tempfile.NamedTemporaryFile("w",
                                    prefix = name,
                                    suffix = ".stan",
                                    dir = job_dir,
                                    delete = False)
    f.write(post.model.code("stan"))
    model_file = f.name
    f.close()

    f = tempfile.NamedTemporaryFile("w",
                                    prefix = name,
                                    suffix = ".dat",
                                    dir = job_dir,
                                    delete = False)
    json.dump(post.data.values(), f, indent = 2, sort_keys = True)
    data_file = f.name
    f.close()

    try:
        cmdstanpy.CmdStanModel(stan_file = model_file)
    except Exception as e:
        print(e)

    manifest["jobs"].append({ model_file : model_file,
                              data_file : data_file })

mf = tempfile.NamedTemporaryFile("w",
                                 prefix = "manifest_",
                                 suffix = ".json",
                                 dir = job_dir,
                                 delete = False)
json.dump(manifest, mf, indent = 2, sort_keys = True)
mf.close()

print("cmdstan dir: ", cmdstan_dir)
print("job dir: ", job_dir)
