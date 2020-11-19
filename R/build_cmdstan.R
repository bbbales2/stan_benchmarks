#!/usr/bin/env Rscript

library(argparse)
parser = ArgumentParser()

parser$add_argument("dir", help = "Folder to install cmdstan in")
parser$add_argument("--cores", type = "integer", default = 1, help = "Number of cores to use")

args = parser$parse_args()

cmdstanr::set_cmdstan_path(args$dir)
cmdstanr::rebuild_cmdstan(dir = args$dir, cores = args$cores)
