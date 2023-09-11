import sys
from path import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import itertools as it

import torch

from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

import matplotlib.dates as mdates
import matplotlib.ticker as ticker


DATA_SUPDIR = Path('/net/projects/schein-lab/jshou/dat/')
RESULTS_SUPDIR = Path('/net/projects/schein-lab/jshou/Posterior_Checks/')
REPLICATION_SUPDIR = Path('/net/projects/schein-lab/jshou/replication/')

def load_team_data(team, dat_supdir=DATA_SUPDIR):
    # load data
    train_pivot = pd.read_csv(dat_supdir.joinpath(team, 'train_pivot.csv'), index_col=0)
    test_pivot = pd.read_csv(dat_supdir.joinpath(team, 'test_pivot.csv'), index_col=0)

    # create a single pivot table for all outcomes
    total_pivot = pd.concat([train_pivot, test_pivot], axis=0)

    # this is a (timestep x county)-array of all outcomes
    all_outcomes_TN = total_pivot.values
    n_timesteps, n_counties = all_outcomes_TN.shape

    # names of the dates and counties
    dates = total_pivot.index
    county_names = total_pivot.columns

    # make sure that the last county is the stadium county
    assert county_names[-1] == 'Stadium_County'

    # the index of the intervention time
    intervention_t = train_pivot.values.shape[0]

    # make sure that the intervention time is correct
    assert np.allclose(all_outcomes_TN[:intervention_t], train_pivot.values)
    assert np.allclose(all_outcomes_TN[intervention_t:], test_pivot.values)

    return total_pivot, intervention_t


def get_reg_arrays(Z, all_outcomes_TN, intervention_t, counterfactual=False, include_previous_outcome=False):
    latent_dim, n_counties = Z.shape
    assert all_outcomes_TN.shape[1] == n_counties
    all_outcomes_NT = all_outcomes_TN.T

    A = np.zeros((n_counties, 1))
    if not counterfactual:
        A[-1] = 1
    
    if include_previous_outcome:
        # Y_{i,t} ~ A + Z_i + Y_{i,-1} 

        B = np.array(all_outcomes_NT[:, intervention_t-1])[:, np.newaxis]
        X = np.concatenate([Z.T, B, A], axis=1)
        assert X.shape == (n_counties, latent_dim + 2)
    else:
        X = np.concatenate([Z.T, A], axis=1)
        assert X.shape == (n_counties, latent_dim + 1)

    Y = np.array(all_outcomes_NT[:, intervention_t:])
    return X, Y


def get_counterfactual_from_best_reg(Z, all_outcomes_TN, intervention_t, include_previous_outcome=True,
                                     reg_type='Ridge', reg_params={"alpha": [0, 1e-4,1e-3, 1e-2]}):
    
    X, Y = get_reg_arrays(Z, all_outcomes_TN, intervention_t, include_previous_outcome=include_previous_outcome)

    if reg_type == 'Ridge':
        reg = Ridge()
    elif reg_type == 'MLP': 
        reg = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=5000)
    elif reg_type == 'Lasso':
        reg = Lasso()
    
    cv = GridSearchCV(reg, reg_params, scoring='r2', cv=5)
    cv_results = cv.fit(X, Y)
    best_reg = cv_results.best_estimator_

    X, _ = get_reg_arrays(Z, all_outcomes_TN, intervention_t, include_previous_outcome=include_previous_outcome, counterfactual=True)
    return best_reg.predict(X)[-1], best_reg  # return only affected county and model


def plot_counterfactual_trajectories(total_pivot, intervention_t, counterfactual_preds, rsc_test_pred, team, fig_path=None):
    all_outcomes_TN = total_pivot.values
    dates = pd.to_datetime(total_pivot.index)

    plt.plot(dates, all_outcomes_TN[:, -1], color='orange', lw=3, label='Factual')
    plt.axvline(dates[intervention_t], color='k', lw=1, linestyle='-', label='Day of intervention')
    
    # Compute percentiles and median across trajectories for each time point
    lower_bound = np.percentile(counterfactual_preds, 5, axis=0)
    upper_bound = np.percentile(counterfactual_preds, 95, axis=0)
    median_trajectory = np.median(counterfactual_preds, axis=0)

    post_dates = dates[intervention_t:]
    plt.fill_between(post_dates, lower_bound, upper_bound, color='blue', alpha=0.1, label='90% posterior credible interval')
    plt.plot(post_dates, median_trajectory, color='blue', linestyle=':', lw=3, label='Posterior median counterfactual')
    plt.plot(post_dates, rsc_test_pred, color='green', linestyle='--', lw=3, label='Robust SC counterfactual')
    plt.legend(fontsize=13)
    plt.title(team, fontsize=17)
    plt.xlabel('Day', fontsize=13)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))  # Every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Month and year
    # Format y-ticks to show 'K' for thousands
    def thousands_formatter(x):
        return '%1.0fK' % (x*1e-3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('Number of new COVID-19 cases', fontsize=13)

    plt.gcf().set_size_inches(10, 6)  # Set the figure size (width, height)
    plt.tight_layout() 
    if fig_path is not None:
        plt.savefig(fig_path, format='pdf', dpi=1000)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=Path, default=DATA_SUPDIR)
    p.add_argument('-r', '--results', type=Path, default=RESULTS_SUPDIR)
    p.add_argument('-o', '--out', type=Path, default=REPLICATION_SUPDIR)
    p.add_argument('--team', type=str, required=True)
    p.add_argument('--model', type=str, required=True)
    p.add_argument('-k', '--latent_dim', type=int, required=True)
    p.add_argument('--reg_type', type=str, default='Ridge', choices=['Ridge', 'Lasso', 'MLPRegressor'])
    p.add_argument('--include_previous_outcome', action="store_true", default=False)
    args = p.parse_args()

    # load data
    total_pivot, intervention_t = load_team_data(args.team, data_supdir=args.data)
    all_outcomes_TN = total_pivot.values
    n_timesteps, n_counties = all_outcomes_TN.shape

    # create output directory
    team_outdir = args.out.joinpath(args.team)
    team_outdir.makedirs_p()

    # load rSC counterfactual predictions
    rsc_pred_file = args.out.joinpath(args.team, 'rSC', 'rsc_pred.npy')
    assert rsc_pred_file.exists(), f"No rSC results for {args.team}"
    rsc_pred = np.load(rsc_pred_file)
    rsc_test_pred = rsc_pred[intervention_t:]

    # load posterior samples
    posterior_samples_file = args.results.joinpath(args.team).files(f'{args.model}_posterior_samples_None_01_latent_dim_{args.latent_dim}.pth')
    assert posterior_samples_file.exists(), f"No posterior samples for {args.team} with {args.model} and K={args.latent_dim}"
    posterior_samples = torch.load(posterior_samples_file)
    Z_samples = np.array(posterior_samples['Z'])
    assert Z_samples[0].shape == (args.latent_dim, n_counties)

    # create directory for model, team, and K
    model_team_k_outdir = team_outdir.joinpath(args.model, f'K_{args.latent_dim}')
    model_team_k_outdir.makedirs_p()

    # save out posterior samples
    np.save(model_team_k_outdir.joinpath('Z_samples.npy'), Z_samples)
    
    # compute counterfactuals for every posterior sample using Ridge regression
    counterfactual_preds = []
    for Z in Z_samples:
        cf_pred, best_reg = get_counterfactual_from_best_reg(Z,
                                                    reg_type=args.reg_type,
                                                    all_outcomes_TN=all_outcomes_TN, 
                                                    intervention_t=intervention_t, 
                                                    include_previous_outcome=args.include_previous_outcome)
        counterfactual_preds.append(cf_pred)
    counterfactual_preds = np.array(counterfactual_preds)

    final_outdir = model_team_k_outdir.joinpath(f'reg_type_{args.reg_type}', f'include_prev_{args.include_previous_outcome}')
    final_outdir.makedirs_p()

    np.save(final_outdir.joinpath('counterfactuals.npy'), counterfactual_preds)

    fig_path = final_outdir.joinpath(f'{args.team}_K-{args.latent_dim}_include-{args.include_previous_outcome}_regtype-{args.reg_type}_plot.pdf')
    plot_counterfactual_trajectories(total_pivot=total_pivot, 
                                     intervention_t=intervention_t, 
                                     counterfactual_preds=counterfactual_preds, 
                                     rsc_test_pred=rsc_test_pred,
                                     team=args.team,
                                     fig_path=fig_path)
    print(fig_path)



