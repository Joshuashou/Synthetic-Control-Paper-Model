import sys
from path import Path
from argparse import ArgumentParser
import subprocess

from tqdm import tqdm

DATA_SUPDIR = Path('/net/projects/schein-lab/jshou/dat/')
RESULTS_SUPDIR = Path('/net/projects/schein-lab/jshou/Posterior_Checks/')
REPLICATION_SUPDIR = Path('/net/projects/schein-lab/jshou/replication/')


def create_bash_script(out_dir, team, model, latent_dim, reg_type, include_previous_outcome):
    job_name = f'{team[:3]}-{latent_dim}-{int(include_previous_outcome)}'
    
    bash_script = \
    f"""#!/bin/bash
    #SBATCH --output={out_dir.joinpath('output.log')}
    #SBATCH --error={out_dir.joinpath('error.log')}
    #SBATCH --job-name={job_name}
    #SBATCH --account=schein
    #SBATCH --partition=general
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --mem-per-cpu=4500
    #SBATCH --time=3:00:00
    #SBATCH --mail-type=FAIL
    #SBATCH --mail-user=schein@uchicago.edu
    # ​
    # source /home/schein/miniconda3/etc/profile.d/conda.sh
    # conda activate /home/schein/miniconda3
    # python deconfound_and_plot.py --out={out_dir} --team={team} --model={model} --latent_dim={latent_dim} --reg_type={reg_type} --include_previous_outcome={include_previous_outcome}
    """
    return bash_script

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

                bash_script = create_bash_script(out_dir=out_dir, 
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
