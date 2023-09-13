import sys
from path import Path
from tqdm import tqdm
from sbatch import create_sbatch, run_sbatch

# get the directory of this script
SRC_DIR = Path(__file__).parent
if '/home/jshou' not in SRC_DIR and '/home/schein' not in SRC_DIR:
    raise NotImplementedError('This script is only configured to run for CNET-IDs: {jshou, schein}.')

# TODO: be more programmatic about this; this is hard-coded for just jshou/schein
CNET_ID = 'jshou' if 'jshou' in SRC_DIR else 'schein'
if CNET_ID == 'schein': 
    PY_SOURCE_STR = '# source /home/schein/miniconda3/etc/profile.d/conda.sh\n# conda activate /home/schein/miniconda3'
elif CNET_ID == 'jshou':
    PY_SOURCE_STR = '# source /home/jshou/miniconda3/etc/profile.d/conda.sh\n# conda activate /home/jshou/miniconda3/envs/sc_env'

PYTHON_EXE_PATH = sys.executable

if __name__ == '__main__':
    SYNTH_DAT_DIR = Path('/net/projects/schein-lab/jshou/synth_dat')
    for data_path in tqdm(SYNTH_DAT_DIR.walkfiles('*train_pivot.csv')):
        data_dir = data_path.parent
        
        data_attrs = Path(data_dir.split(SYNTH_DAT_DIR)[1]).splitall()[1:-1]
        team, data_seed = data_attrs[0], data_attrs[-1]
        data_attrs = data_attrs[1:-1]
        data_attr_str = '-'.join([attr.split('_')[-1] for attr in data_attrs])
        job_name = f'{team[:4]}-{data_attr_str}'
        
        for model in ['GAP']:
            for latent_dim in [10]:
                for seed in [617]:
                    out_dir = data_dir.joinpath('results', model, f'latent_dim_{latent_dim}', f'model_seed_{seed}')
                    out_dir.makedirs_p()

                    team = Path(data_path.split(SYNTH_DAT_DIR)[1]).splitall()[1]

                    script_args = [] 
                    script_kwargs = {
                         'data': data_path,
                         'out': out_dir,
                         'model': model,
                         'model_seed': seed,
                         'latent_dim': latent_dim
                    }

                    sbatch_kwargs = {
                        'partition': 'general',
                        'nodes': 1,
                        'ntasks': 1,
                        'mem-per-cpu': 4500,
                        'time': '3:00:00',
                        'mail-type': 'FAIL'
                    }

                    sbatch_script = create_sbatch(SRC_DIR.joinpath('run_semi_synthetic_experiment.py'), 
                                                  output_dir=out_dir, 
                                                  script_exe=PYTHON_EXE_PATH,
                                                  script_args=script_args, 
                                                  script_kwargs=script_kwargs, 
                                                  job_name=job_name, 
                                                  cnet_id=CNET_ID, 
                                                  sbatch_kwargs=sbatch_kwargs,
                                                  source_str=PY_SOURCE_STR)

                    with open(out_dir.joinpath('sbatch.sh'), 'w') as f:
                        f.write(sbatch_script)
                    print(out_dir.joinpath('sbatch.sh'))

                    result = run_sbatch(sbatch_script)
                    sys.exit()