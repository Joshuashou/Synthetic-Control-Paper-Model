import numpy as np
import numpy.random as rn
import pandas as pd

import torch
import pyro

import sys
from path import Path
from argparse import ArgumentParser

from bayesian_factor_model import PPCA, GAP, run_NUTS_with_mask


def main(args=None):
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=Path, required=True)
    p.add_argument('-m', '--mask', type=Path, default=None)
    p.add_argument('-o', '--out', type=Path, default=None)
    p.add_argument('--model', type=str, default='GAP', choices=['GAP', 'PPCA'])
    p.add_argument('--model_seed', type=int, default=None)
    p.add_argument('-k', '--latent_dim', type=int, default=10)

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

    # create output directory
    out_dir = args.out
    if out_dir is None:
        out_dir = args.data.parent
        out_dir = out_dir.joinpath('results', args.model, f'latent_dim{args.latent_dim}', f'model_seed_{seed}')
    out_dir.makedirs_p()

    # load data
    train_pivot = pd.read_csv(args.data, index_col=0)
    train_data = torch.tensor(train_pivot.values)

    # load mask
    mask = args.mask
    if mask is not None:
        mask = torch.load(args.mask)

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


if __name__ == '__main__':
    main()



