import numpy as np
import numpy.random as rn

import torch

import pyro
import pyro.distributions as dist
import pyro.infer.mcmc as mcmc

from tqdm import tqdm


class PPCA:
    """Probabilistic PCA (PPCA)"""

    def __init__(self, latent_dim, variance_support=(0, 10)):
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
        with pyro.plate("data"):
            likelihood = self.likelihood(H=H, Z=Z)
            if mask is not None:
                likelihood = likelihood.mask(mask)
            Y = pyro.sample("Y", likelihood, obs=data)
        
        return Y, Z, H
    
    def gibbs_sample(self, data, mask=None, num_samples=2000, warmup_steps=2000):
        n_time, n_unit = data.shape
        shp, rte, latent_dim = self.shp, self.rte, self.latent_dim

        data = data.detach().numpy().copy()
        mask = mask.detach().numpy().copy()
        data[mask] = 0

        # Initialize latent factors from prior (K x N)
        Z = np.zeros((warmup_steps + num_samples, latent_dim, n_unit))
        Z[0] = rn.gamma(shp, 1./rte, size=(latent_dim, n_unit)) 

        # Sample weights from prior (T x K)
        H = np.zeros((warmup_steps + num_samples, n_time, latent_dim))
        H[0] = rn.gamma(shp, 1./rte, size=(n_time, latent_dim)) 

        rn = np.random.default_rn()
        for s in tqdm(range(1, warmup_steps + num_samples)):
            # Allocation step
            P_TNK = np.einsum('tk,nk->tnk', H[s-1], Z[s-1])
            P_TNK /= np.sum(P_TNK, axis=2, keepdims=True)
            Y_TNK = rn.multinomial(data, P_TNK)

            # Update latent factors
            post_shp = shp + Y_TNK.sum(axis=1)
            post_rte = rte + np.einsum('tn,kn->tk', mask, Z[s-1])
            H[s] = rn.gamma(post_shp, 1. / post_rte)

            post_shp = shp + Y_TNK.sum(axis=0).T
            post_rte = rte + np.einsum('tn,kn->kt', mask, H[s])
            Z[s] = rn.gamma(post_shp, 1. / post_rte)
        
        posterior_samples = {'Z': torch.from_numpy(Z[warmup_steps:]), 
                             'H': torch.from_numpy(H[warmup_steps:])}
        return posterior_samples


def run_NUTS_with_mask(model, data, mask, warmup_steps, num_samples):
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


def create_mask(data, mask_type="plaid"):
    if not mask_type.lower() in ["plaid", "random", "speckled", "strong", "end"]:
        print("Warning: Mask type not recognized, defaulting to no mask.")
    mask_type = str(mask_type).lower()

    n_time, n_unit = data.shape
    mask = torch.ones(n_time, n_unit)

    if mask_type == ["plaid", "random"]:
        mask_rows = torch.randperm(n_time)[:20]
        mask_cols = torch.randperm(n_unit)[:5]
        for t in mask_rows:
            for i in mask_cols:
                mask[t, i] = 0
    
    elif mask_type == "speckled":
        # hold out 1% of the data
        mask = torch.bernoulli(mask * 0.99)
        
    elif mask_type in ["strong", "end"]:
        mask[-30:, -3:] = 0
    
    mask = mask.bool()
    return mask


def population_predictive_check(data, mask, model, posterior_samples):
    model_name = model.__class__.__name__
    posterior_predictive_samples = model.likelihood(**posterior_samples).sample()

    log_prob_fake = model.likelihood(**posterior_samples).log_prob(posterior_predictive_samples)
    log_prob_true = model.likelihood(**posterior_samples).log_prob(data)

    d_fake = -log_prob_fake[:, ~mask].sum(axis=-1)
    d_true = -log_prob_true[:, ~mask].sum(axis=-1)

    ppop = (d_fake > d_true).float().mean()
    return ppop