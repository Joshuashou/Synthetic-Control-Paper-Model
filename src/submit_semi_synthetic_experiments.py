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
    PY_SOURCE_STR = 'source /home/schein/miniconda3/etc/profile.d/conda.sh\nconda activate /home/schein/miniconda3'
elif CNET_ID == 'jshou':
    PY_SOURCE_STR = 'source /home/jshou/miniconda3/etc/profile.d/conda.sh\nconda activate /home/jshou/miniconda3/envs/sc_env'

PYTHON_EXE_PATH = sys.executable

sbatch_kwargs = {
    'partition': 'general',
    'nodes': 1,
    'ntasks': 1,
    'mem-per-cpu': 4500,
    'time': '3:00:00',
    'mail-type': 'FAIL'
}

if __name__ == '__main__':
    # TODO: be more programmatic about this
    TEAMS = ['Philadelphia', 'Minnesota', 'Baltimore', 'New Orleans']

    SYNTH_DAT_DIR = Path('/net/projects/schein-lab/jshou/synth_dat/new_dat_nov_20_2023')

    # for data_path in tqdm(SYNTH_DAT_DIR.walkfiles('*train_pivot.csv')):
    #     data_dir = data_path.parent
        # data_attrs = Path(data_dir.split(SYNTH_DAT_DIR)[1]).splitall()[1:-1]
        # team, data_seed = data_attrs[0], data_attrs[-1]
        # data_attrs = data_attrs[1:-1]
        # data_attr_str = '-'.join([attr.split('_')[-1] for attr in data_attrs])

    for team in TEAMS:
        for subdir in SYNTH_DAT_DIR.joinpath(team).dirs():
            for latent_dim in [5, 10, 15]:
                ### Run robust SC
                model = 'rSC'
                job_name = f'{team[:4]}-{model}--{latent_dim}'
        
                script_args = [] 
                script_kwargs = {
                        'data': subdir,
                        'out': None,
                        'model': 'rSC',
                        'model_seed': 617,
                        'latent_dim': latent_dim
                }

                sbatch_script = create_sbatch(SRC_DIR.joinpath('run_semi_synthetic_experiment.py'), 
                                              output_dir=subdir, 
                                              script_exe='python',
                                              script_args=script_args, 
                                              script_kwargs=script_kwargs, 
                                              job_name=job_name, 
                                              cnet_id=CNET_ID, 
                                              sbatch_kwargs=sbatch_kwargs,
                                              source_str=PY_SOURCE_STR)

                with open(subdir.joinpath('sbatch.sh'), 'w') as f:
                    f.write(sbatch_script)
                print(subdir.joinpath('sbatch.sh'))

                result = run_sbatch(sbatch_script)

                # Now run all the Bayesian models (comment in to run)

                # for model in ['GAP', 'PPCA']:
                #     for seed in [617, 781]:
                #         out_dir = data_dir.joinpath('results', model, f'latent_dim_{latent_dim}', f'model_seed_{seed}')
                #         out_dir.makedirs_p()
                #         if out_dir.joinpath('posterior_samples.npz').exists():
                #             continue

                #         script_args = [] 
                #         script_kwargs = {
                #              'data': data_path,
                #              'out': out_dir,
                #              'model': model,
                #              'model_seed': seed,
                #              'latent_dim': latent_dim
                #         }

                #         sbatch_script = create_sbatch(SRC_DIR.joinpath('run_semi_synthetic_experiment.py'), 
                #                                       output_dir=out_dir, 
                #                                       script_exe='python',
                #                                       script_args=script_args, 
                #                                       script_kwargs=script_kwargs, 
                #                                       job_name=job_name, 
                #                                       cnet_id=CNET_ID, 
                #                                       sbatch_kwargs=sbatch_kwargs,
                #                                       source_str=PY_SOURCE_STR)

                #         with open(out_dir.joinpath('sbatch.sh'), 'w') as f:
                #             f.write(sbatch_script)
                #         print(out_dir.joinpath('sbatch.sh'))

                #         result = run_sbatch(sbatch_script)