import sys
from path import Path
from argparse import ArgumentParser
import subprocess

from tqdm import tqdm

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

    sbatch_script = \
    f"""#!/bin/bash
    #SBATCH --output={output_dir.joinpath('output.log')}
    #SBATCH --error={output_dir.joinpath('error.log')}
    #SBATCH --job-name={job_name}
    #SBATCH --account={cnet_id}
    #SBATCH --mail-user={cnet_id}@uchicago.edu
    {sbatch_kwarg_str}
    # ​
    {source_str}
    # {script_exe} {script_path} {script_arg_str} {script_kwarg_str}
    """
    return sbatch_script

def run_sbatch(sbatch_script, verbose=True):
    # Pipe script to sbatch
    result = subprocess.run(["sbatch"], input=sbatch_script, text=True, capture_output=True)
    if verbose:
        print(result.stdout)
    return result

if __name__ == '__main__':
    for posterior_sample_file in RESULTS_SUPDIR.walkfiles(f'*posterior_samples_*.pth'):
        team = str(posterior_sample_file.parent.basename())
        if team != 'Seattle':
            continue

        model = posterior_sample_file.basename().split('_')[0]
        latent_dim = int(posterior_sample_file.basename().split('_')[-1].split('.')[0])

        for include_previous_outcome in [True, False]:
            for reg_type in ['Ridge']:
                out_dir = REPLICATION_SUPDIR.joinpath(team, model, f'K_{latent_dim}', f'reg_type_{reg_type}', f'include_prev_{include_previous_outcome}')
                out_dir.makedirs_p()

                script_args = [] 
                if include_previous_outcome:
                    script_args += ['--include_previous_outcome']
                
                script_kwargs = {
                    '--out': out_dir,
                    '--team': team,
                    '--model': model,
                    '--latent_dim': latent_dim,
                    '--reg_type': reg_type
                    }

                bash_script = create_sbatch(script_path=SRC_DIR.joinpath('deconfound_and_plot.py'),
                                            script_args=script_args,
                                            script_kwargs=script_kwargs,
                                            lang='python',
                                            job_name=f'{team[:3]}_{model}_{latent_dim}_{reg_type}_{include_previous_outcome}',
                                            cnet_id='schein',
                                            sbatch_kwargs=DEFAULT_SBATCH_KWARGS
                    
                    
                    
                    out_dir=out_dir, 
                                                 team=team, 
                                                 model=model, 
                                                 latent_dim=latent_dim, 
                                                 reg_type=reg_type, 
                                                 include_previous_outcome=include_previous_outcome)
                # print(bash_script)

                with open('temp_script.sh', 'w') as f:
                    f.write(bash_script)
                # # Pipe script to sbatch
                # result = subprocess.run(["sbatch"], input=bash_script, text=True, capture_output=True)
                # print(result.stdout)
                sys.exit()
