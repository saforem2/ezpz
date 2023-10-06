#!/bin/bash --login

function getSlurmEnv() {
  VARS=$(env | grep SLU)
}

function saveSlurmEnv() {
  outfile="~/.slurmjob"
  echo "Saving SLURM_* to ~/.slurmjob"
  echo "${VARS[*]}" > ~/.slurmjob
  sed -i 's/^SLURM/export\ SLURM/g' ~/.slurmjob
  sed -i 's/(x2)//g' ~/.slurmjob
}


getSlurmEnv
saveSlurmEnv
