import sys
from path import Path
import subprocess

from argparse import ArgumentParser

# get the directory of this script
SRC_DIR = Path(__file__).parent
if '/home/jshou' not in SRC_DIR and '/home/schein' not in SRC_DIR:
    raise NotImplementedError('This script is only configured to run for CNET-IDs: {jshou, schein}.')

# get full path to the python build that is executing this script
PY_EXEC_STR = sys.executable

# TODO: be more programmatic about this; this is hard-coded for just jshou/schein
CNET_ID = 'jshou' if 'jshou' in SRC_DIR else 'schein'
if CNET_ID == 'schein': 
    PY_SOURCE_STR = '# source /home/schein/miniconda3/etc/profile.d/conda.sh\n# conda activate /home/schein/miniconda3'
elif CNET_ID == 'jshou':
    PY_SOURCE_STR = '# source /home/jshou/miniconda3/etc/profile.d/conda.sh\n# conda activate /home/jshou/miniconda3/envs/sc_env'


DEFAULT_SBATCH_KWARGS = {
    'partition': 'general',
    'nodes': 1,
    'ntasks': 1,
    'mem-per-cpu': 4500,
    'time': '3:00:00',
    'mail-type': 'FAIL'
}

def create_sbatch(script_path, output_dir=Path('.'), script_exe=PY_EXEC_STR, script_args=[], script_kwargs={}, 
                  job_name='', cnet_id=CNET_ID, sbatch_kwargs=DEFAULT_SBATCH_KWARGS, source_str=PY_SOURCE_STR):
    script_arg_str = ' '.join(f'--{v}' for v in script_args)
    script_kwarg_str = ' '.join(f'--{k}={v}' for k, v in script_kwargs.items())
    sbatch_kwarg_str = '\n'.join(f'#SBATCH --{k}={v}' for k, v in sbatch_kwargs.items())

    sbatch_script_lines = [
        "#!/bin/bash",
        f"#SBATCH --output={output_dir.joinpath('output.log')}",
        f"#SBATCH --error={output_dir.joinpath('error.log')}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={cnet_id}",
        f"#SBATCH --mail-user={cnet_id}@uchicago.edu",
        sbatch_kwarg_str,
        source_str,
        f"# {script_exe} {script_path} {script_arg_str} {script_kwarg_str}"
    ]
    sbatch_script = '\n'.join(sbatch_script_lines)
    return sbatch_script

def run_sbatch(sbatch_script, verbose=True):
    # Pipe script to sbatch
    result = subprocess.run(["sbatch"], input=sbatch_script, text=True, capture_output=True)
    if verbose:
        print(result.stdout)
    return result

