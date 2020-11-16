#!/usr/bin/env Rscript

## Stolen from https://github.com/stan-dev/cmdstanr/issues/300#issuecomment-702824987
library(argparse)
parser = ArgumentParser()

parser$add_argument("dir", help = "Folder to install cmdstan in")
parser$add_argument("--cores", type = "integer", default = 1, help = "Number of cores to use")
parser$add_argument("--cmdstan_branch", default = "develop", help = "cmdstan repo branch")
parser$add_argument("--stan_branch", default = "develop", help = "stan repo branch")
parser$add_argument("--math_branch", default = "develop", help = "math repo branch")
parser$add_argument("--cmdstan_url", default = "http://github.com/stan-dev/cmdstan", help = "cmdstan repo url")
parser$add_argument("--stan_url", default = "http://github.com/stan-dev/stan", help = "stan repo url")
parser$add_argument("--math_url", default = "http://github.com/stan-dev/math", help = "math repo url")

args = parser$parse_args()

if (!dir.exists(args$dir)) {
    dir.create(args$dir)
}

git2r::clone(url = args$cmdstan_url,
             branch = args$cmdstan_branch,
             local_path = file.path(args$dir))

git2r::clone(url = args$stan_url,
             branch = args$stan_branch,
             local_path = file.path(args$dir, "stan"))

git2r::clone(url = args$math_url,
             branch = args$math_branch,
             local_path = file.path(args$dir, "stan", "lib", "stan_math"))

cmdstanr::install_cmdstan(dir = args$dir, cores = args$cores)
