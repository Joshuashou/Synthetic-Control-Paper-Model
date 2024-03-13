import numpy as np
import numpy.random as rn
import pandas as pd

import torch
import pyro

import sys
from path import Path
from argparse import ArgumentParser

from bayesian_factor_model import PPCA, GAP, run_NUTS_with_mask
from deconfound_and_plot import get_counterfactual_from_best_reg

from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.tests import testdata


def main(args=None):
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=Path, required=True)
    p.add_argument('-m', '--mask', type=Path, default=None)
    p.add_argument('-o', '--out', type=Path, default=None)
    p.add_argument('--model', type=str, default='GAP', choices=['GAP', 'PPCA', 'rSC'])
    p.add_argument('--model_seed', type=int, default=None)
    p.add_argument('-k', '--latent_dim', type=int, default=10)
    p.add_argument('--reg_type', type=str, default='Ridge', choices=['Ridge', 'Lasso', 'MLPRegressor'])

    if args is None:
        args = sys.argv[1:]
    args = p.parse_args(args)

    # create and set random seeds
    seed = args.model_seed
    if seed is None:
        seed = np.random.randint(0, 1000)
    rn.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)

    if args.data.isfile():
        train_pivot_files = [args.data]
    else:
        train_pivot_files = args.data.walkfiles('*train_pivot.csv')

    for train_pivot_file in train_pivot_files:
        # create output directory
        out_dir = args.out
        if out_dir is None or out_dir == 'None':
            out_dir = train_pivot_file.parent
            out_dir = out_dir.joinpath('results', args.model, f'latent_dim_{args.latent_dim}', f'model_seed_{seed}')
        out_dir.makedirs_p()

        train_pivot = pd.read_csv(train_pivot_file, index_col=0)
        train_data = torch.tensor(train_pivot.values)

        # load test data
        test_pivot = pd.read_csv(train_pivot_file.parent.joinpath('test_pivot.csv'), index_col=0)
        total_pivot = pd.concat([train_pivot, test_pivot], axis=0)
        intervention_t = train_pivot.values.shape[0]

        if args.model in ['GAP', 'PPCA']:
            # load mask
            mask = args.mask
            if mask is not None:
                mask = torch.load(args.mask)

            if not out_dir.joinpath('posterior_samples.npz').exists():
                # setup model
                if args.model == 'GAP':
                    model = GAP(latent_dim=args.latent_dim, shp=1.0, rte=1.0)
                    
                elif args.model == 'PPCA':
                    model = PPCA(latent_dim=args.latent_dim, variance_support=(0, 10))

                # run MCMC
                posterior_samples = run_NUTS_with_mask(model=model.model, 
                                                       data=train_data,
                                                       mask=mask,
                                                       warmup_steps=1000,
                                                       num_samples=2000)
                
                # covert to numpy
                posterior_samples = {k: v.numpy() for k, v in posterior_samples.items()}

                # save using np.savez_compressed
                np.savez_compressed(out_dir.joinpath('posterior_samples.npz'), **posterior_samples)
                print(out_dir.joinpath('posterior_samples.npz'))
            else:
                posterior_samples = np.load(out_dir.joinpath('posterior_samples.npz'))
            


            # compute counterfactuals for every posterior sample using Ridge regression
            Z_samples = posterior_samples['Z']
            func = lambda z: get_counterfactual_from_best_reg(z,
                                                            total_pivot=total_pivot, 
                                                            intervention_t=intervention_t, 
                                                            reg_type=args.reg_type,
                                                            include_previous_outcome=True)[0]
            
            counterfactual_preds = np.array([func(Z) for Z in Z_samples])
            np.save(out_dir.joinpath('counterfactuals.npy'), counterfactual_preds) 
        
        elif args.model == 'rSC':
            
            assert 'Stadium_County' in train_pivot.columns
            rscModel = RobustSyntheticControl(seriesToPredictKey='Stadium_County', 
                                            kSingularValuesToKeep=args.latent_dim, 
                                            M=len(train_pivot), 
                                            probObservation=1.0, 
                                            modelType='svd', 
                                            svdMethod='numpy', 
                                            otherSeriesKeysArray=[col for col in train_pivot.columns if col != 'Stadium_County'])

            rscModel.fit(train_pivot)
            counterfactual_preds = test_pivot.values[:, :-1] @ rscModel.model.weights
            np.save(out_dir.joinpath('counterfactuals.npy'), counterfactual_preds) 

if __name__ == '__main__':
    main()



