import numpy as np
import numpy.random as rn
import pandas as pd

import torch
import pyro
import pyro.distributions as dist
import pyro.infer.mcmc as mcmc

import sys
from path import Path
from argparse import ArgumentParser


class PPCA:
    """Probabilistic PCA (PPCA)"""

    def __init__(self, latent_dim, variance_support=(0, 20)):
        self.latent_dim = latent_dim
        self.variance_support = variance_support
    
    def likelihood(self, **kwargs):
        H = kwargs['H']
        Z = kwargs['Z']
        # this indexing handles the possibility of posterior samples
        sigma = kwargs['sigma'][..., None, None]
        return dist.Normal(H @ Z, sigma)
    
    def model(self, data, mask=None):
        n_time, n_unit = data.shape

        # Sample latent factors (K x N)
        Z = pyro.sample('Z', dist.Normal(torch.zeros(self.latent_dim, n_unit), 1.).to_event(2))

        # Sample weights (T x K)
        H = pyro.sample('H', dist.Normal(torch.zeros(n_time, self.latent_dim), 1.).to_event(2))

        # Sample observed variance
        sigma = pyro.sample("sigma", dist.Uniform(self.variance_support[0], self.variance_support[1]))

        # Sample observed data
        with pyro.plate("data"):
            likelihood = self.likelihood(H=H, Z=Z, sigma=sigma)
            if mask is not None:
                likelihood = likelihood.mask(mask)
            Y = pyro.sample("Y", likelihood, obs=data)

        return Y, Z, H, sigma

class GAP:
    """The Gamma-Poisson (GaP) factor model"""

    def __init__(self, latent_dim, shp=1., rte=1.):
        self.latent_dim = latent_dim
        self.shp = shp
        self.rte = rte
    
    def likelihood(self, **kwargs):
        return dist.Poisson(kwargs['H'] @ kwargs['Z'])
    
    def model(self, data, mask=None):
        n_time, n_unit = data.shape
        shp, rte, latent_dim = self.shp, self.rte, self.latent_dim

        # Sample latent factors (K x N)
        Z = pyro.sample("Z", dist.Gamma(concentration=shp, rate=rte).expand([latent_dim, n_unit]).to_event(2))

        # Sample weights (T x K)
        H = pyro.sample("H", dist.Gamma(concentration=shp, rate=rte).expand([n_time, latent_dim]).to_event(2))

        # Sample observed data
        likelihood = self.likelihood(H=H, Z=Z)
        if mask is not None:
            likelihood = likelihood.mask(mask)
        Y = pyro.sample("Y", likelihood, obs=data)
        
        return Y, Z, H


def run_NUTS_with_mask(model, data, mask=None, warmup_steps=1000, num_samples=2000):
    """Runs NUTS with given model, data, mask and returns posterior samples."""
    
    pyro.clear_param_store() # do we need this?

    # Define MCMC kernel
    kernel = mcmc.NUTS(model)
    mcmc_run = mcmc.MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps)

    # Run MCMC process on our data with given latent dimension and mask
    mcmc_run.run(data, mask)

    # Extract samples
    posterior_samples = mcmc_run.get_samples()
    
    return posterior_samples


def main(args=None):
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=Path, required=True)
    p.add_argument('-m', '--mask', type=Path, default=None)
    p.add_argument('-o', '--out', type=Path, default=None)
    p.add_argument('--model', type=str, default='GAP', choices=['GAP', 'PPCA'])
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('-k', '--latent_dim', type=int, default=10)

    if args is None:
        args = sys.argv[1:]
    args = p.parse_args(args)

    # create and set random seeds
    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, 1000)
    rn.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)

    # create output directory
    out_dir = args.out
    if out_dir is None:
        out_dir = args.data.parent
    out_dir = out_dir.joinpath('results', args.model, f'latent_dim{args.latent_dim}', f'seed_{seed}')
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
        
    elif args.model == "PPCA":
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
    for data_path in Path('/net/projects/schein-lab/jshou/synth_dat').walkfiles('*train_pivot.csv'):
        args = ["-d", data_path, "--seed", "617"]
        main(args=args)