# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plot temperature response over time

# %% [markdown]
# ## Method: 
#

# %% [markdown]
# ## IRF:
# Using forcings from RCMIP models and the impulse response function:
# \begin{align*}
# \text{IRF}(t)=& 0.885\cdot (\frac{0.587}{4.1}\cdot exp(\frac{-t}{4.1}) + \frac{0.413}{249} \cdot exp(\frac{-t}{249}))\\
# \text{IRF}(t)= &  \sum_{i=1}^2\frac{\alpha \cdot c_i}{\tau_i}\cdot exp\big(\frac{-t}{\tau_1}\big) 
# \end{align*}
# with $\alpha = 0.885$, $c_1=0.587$, $\tau_1=4.1$, $c_2=0.413$ and $\tau_2 = 249$.

# %% Thus we can estimate the mean surface temperature change from some reference year (here 0) by using [markdown]
# the estimated ERF$_x$ for some forcing agent $x$ as follows: 

# %% [markdown]
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %% [markdown]
# # Code + figures

# %% [markdown]
# ## Imports:

# %%
import xarray as xr
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# %load_ext autoreload
# %autoreload 2

# %%
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

PATH_DT = OUTPUT_DATA_DIR + '/dT_data_rcmip_models.nc'

# %%
FIGURE_DIR = RESULTS_DIR + '/figures/'

# %%

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'

# %% [markdown]
# ### Define variables to look at:

# %%
from ar6_ch6_rcmipfigs.utils.misc_func import new_varname

# variables to plot:
variables_erf_comp = [
    'Effective Radiative Forcing|Anthropogenic|CH4',
    'Effective Radiative Forcing|Anthropogenic|Aerosols',
    'Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone',
    'Effective Radiative Forcing|Anthropogenic|F-Gases|HFC',
    'Effective Radiative Forcing|Anthropogenic|Other|BC on Snow']
# total ERFs for anthropogenic and total:
variables_erf_tot = ['Effective Radiative Forcing|Anthropogenic',
                     'Effective Radiative Forcing']
# Scenarios to plot:
scenarios_fl = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF-aerchemmip',
                # 'ssp370-lowNTCF', Due to mistake here
                'ssp585', 'historical']
scenarios_nhist = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF-aerchemmip',
                'ssp370-lowNTCF-gidden',
                   # 'ssp370-lowNTCF', Due to mistake here
                   'ssp585']  # list(set(scenarios_fl)- {'historical'})
climatemodels_fl = ['Cicero-SCM', 'Cicero-SCM-ECS3', 'FaIR-1.5-DEFAULT', 'MAGICC7.1.0.beta-rcmip-phase-1', 'OSCARv3.0']

# List of delta T for variables
name_deltaT = 'Delta T'
variables_dt_comp = [new_varname(var, name_deltaT) for var in variables_erf_comp]


# %%
# Years to plot:
years = ['2040', '2100']

# scn_trans = [trans_scen2plotlabel(label) for label in scenarios_nhist]
# scn_trans = scenarios_nhist  # scenarios except historical

variables_tot = ['Total']
variables_sum = ['Sum SLCFs']


def setup_table_prop(scenario_n='', years=None, vars=None, scens=None):
    if vars is None:
        vars = [var.split('|')[-1] for var in variables_erf_comp]
    if years is None:
        years = ['2040', '2100']
    if scens is None:
        scens = scenarios_nhist
    its = [years, vars]
    _i = pd.MultiIndex.from_product(its, names=['', ''])
    table = pd.DataFrame(columns=scens, index=_i)  # .transpose()
    table.index.name = scenario_n
    return table


# %% [markdown]
# ## Open dataset:

# %% [markdown]
# ### Integrate:
# The code below opens the file generated in [2_compute_delta_T.ipynb](2_compute_delta_T.ipynb) by integrating

# %% [markdown]
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %% [markdown]
# where IRF is the impulse response function and ERF is the effective radiative forcing from RCMIP. 

# %%
ds_DT = xr.open_dataset(PATH_DT)


# %% [markdown]
# ## Compute sum of all SLCF forcers

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_cmap_dic, trans_scen2plotlabel, get_scenario_c_dic, get_scenario_ls_dic

# %%

# ds_DT = dic_ds[0.885]
s_y = '1850'
# cdic = get_scenario_c_dic()

cdic = get_scenario_c_dic()  # get_cmap_dic(ds_DT[scenario].values)
lsdic = get_scenario_ls_dic()  # _scget_ls_dic(ds_DT[climatemodel].values)


def sum_name(var): return '|'.join(var.split('|')[0:2]) + '|' + 'All'


var = variables_erf_comp[0]
f_totn = sum_name(var)
dt_totn = sum_name(new_varname(var, name_deltaT))

# make xarray with variable as new dimension:
_lst_f = []
_lst_dt = []
for var in variables_erf_comp:
    _lst_f.append(ds_DT[var])
    _lst_dt.append(ds_DT[new_varname(var, name_deltaT)])
erf_all = sum_name('Effective Radiative Forcing|Anthropogenic|all')
dt_all = sum_name(new_varname('Effective Radiative Forcing|Anthropogenic|all', name_deltaT))
ds_DT[erf_all] = xr.concat(_lst_f, pd.Index(variables_erf_comp, name='variable'))
ds_DT[dt_all] = xr.concat(_lst_dt, pd.Index(variables_erf_comp, name='variable'))

# %%
ref_year = '2021'
scntab_dic = {}


# tab_tot = setup_table2()
# tab_tot_sd = setup_table2()
def table_of_sts(ds_DT, scenarios_nhist, variables, tab_vars, years, ref_year, sts='mean'):
    """
    Creates pandas dataframe of statistics (mean, median, standard deviation) for change
    in temperature Delta T since year (ref year) for each scenario in scenarios,

    :param ds_DT:
    :param scenarios_nhist:
    :param variables:
    :param tab_vars:
    :param years:
    :param ref_year:
    :param sts:
    :return:
    """
    tabel = setup_table_prop(years=years, vars=tab_vars)
    for scn in scenarios_nhist:
        for var, tabvar in zip(variables, tab_vars):
            dtvar = new_varname(var, name_deltaT) # if ERF name, changes it here.
            tabscn = scn  # Table scenario name the same.
            for year in years:
                _da =ds_DT[dtvar].sel(scenario=scn)
                _da_refy = _da.sel(time=slice(ref_year, ref_year)).squeeze() # ref year value
                _da_y = _da.sel(time=slice(year, year)) # year value
                _tab_da = _da_y - _da_refy
                #_tab_da = ds_DT[dtvar].sel(scenario=scn, time=slice(year, year)) - ds_DT[dtvar].sel(scenario=scn,
                #                                                                                    time=slice(ref_year,
                #                                                                                               ref_year)).squeeze()

                # Do statistics over RCMIP models
                if sts == 'mean':
                    tabel.loc[(year, tabvar), tabscn] = _tab_da.mean('climatemodel').values[0]
                if sts == 'median':
                    tabel.loc[(year, tabvar), tabscn] = _tab_da.median('climatemodel').values[0]
                elif sts == 'std':
                    tabel.loc[(year, tabvar), tabscn] = _tab_da.std('climatemodel').values[0]

    return tabel


# Statistics on Delta T anthropogenic
# Mean
tabel_dT_anthrop = table_of_sts(ds_DT, scenarios_nhist, ['Delta T|Anthropogenic'], ['Total'], years, ref_year)
# Standard deviation
tabel_dT_anthrop_SD = table_of_sts(ds_DT, scenarios_nhist, ['Delta T|Anthropogenic'], ['Total'], years, ref_year, sts='std')
# Mean:
tabel_dT_slcfs = table_of_sts(ds_DT, scenarios_nhist, variables_dt_comp, [var.split('|')[-1] for var in variables_dt_comp], years,
                              ref_year)
# Standard deviation
tabel_dT_slcfs_DF = table_of_sts(ds_DT, scenarios_nhist, variables_dt_comp, [var.split('|')[-1] for var in variables_dt_comp],
                                 years, ref_year, sts='std')
# Compute sum of SLCFs
_ds = ds_DT.copy()
vall = 'Delta T|Anthropogenic|All'
_ds[vall] = _ds[vall].sum('variable')
tabel_dT_sum_slcf = table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year)
tabel_dT_sum_slcf_SD = table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year, sts='std')

# %%
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, len(years), figsize=[10, 8], sharey=False)
tits = ['Near Term surface temperature change (2040 relative to 2021)',
        'Long Term surface T change 2100 relatie to 2021)']
tits = ['Change in GMST in 2040 relative to 2021', 'Change in GMST in 2100 relative to 2021']
for yr, ax, tit in zip(years, axs, tits):
    ntot = 'Scenario total'
    tot_yr = tabel_dT_anthrop.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    tot_sd_yr = tabel_dT_anthrop_SD.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # l =ax.bar(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2, yerr=tab_tot_sd)
    sum_yr = tabel_dT_sum_slcf.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    sum_sd_yr = tabel_dT_sum_slcf_SD.loc[yr].rename(
        {'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    ax.bar(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2,
           yerr=tot_sd_yr.transpose()[ntot].values,
           error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ntot = 'Sum SLCFs'
    # ax.bar(sum_yr.transpose().index, sum_yr.transpose()[ntot].values, color='r', label=ntot, alpha=.2, yerr=sum_sd_yr.transpose()[ntot].values,
    #      error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=1))

    s_x = sum_yr.transpose().index
    s_y = sum_yr.transpose()[ntot].values
    s_err = sum_sd_yr.transpose()[ntot].values
    ax.errorbar(s_x, s_y, color='k', fmt='d', label=ntot, yerr=s_err, linestyle="None")  # ,
    # error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=1))

    _tab = tabel_dT_slcfs.loc[yr].transpose().rename({'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})

    a = _tab.plot(kind='bar', stacked=True, ax=ax, legend=(yr != '2040'))  # , grid=True)#stac)
    if not yr == '2040':
        ax.legend()  # [l],labels=['Sce!!nario total'], loc = 4)#'lower right')

    ax.axhline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_ylabel('$\Delta$ GMST ($^\circ$C)')
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.grid(axis='y', which='major')

fn = RESULTS_DIR + '/figures/stack_bar_influence_years.png'
plt.tight_layout()
ax = plt.gca()

ax.tick_params(axis='y', which='minor')  # ,bottom='off')
plt.savefig(fn, dpi=300)

# %% [markdown]
# ## Error bars only from model uncertainty

# %%
from matplotlib import transforms
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, len(years), figsize=[12, 6], sharex=False, sharey=True)
tits = ['Near Term surface temperature change (2040 relative to 2021)',
        'Long Term surface T change 2100 relatie to 2021)']
tits = ['Change in GMST in 2040 relative to 2021', 'Change in GMST in 2100 relative to 2021']
for yr, ax, tit in zip(years, axs, tits):
    ntot = 'Scenario total'
    # Pick out year and do various renames:
    # Total antropogenic
    tot_yr = tabel_dT_anthrop.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    tot_sd_yr = tabel_dT_anthrop_SD.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # Sum SLCFs
    sum_yr = tabel_dT_sum_slcf.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    sum_sd_yr = tabel_dT_sum_slcf_SD.loc[yr].rename(
        {'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # Plot bars for anthropopogenic total:
    ax.barh(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2,
            xerr=tot_sd_yr.transpose()[ntot].values,
            error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    # Plot bars for SLCFs total:
    ntot = 'Sum SLCFs'
    s_x = sum_yr.transpose().index
    s_y = sum_yr.transpose()[ntot].values
    s_err = sum_sd_yr.transpose()[ntot].values
    ax.errorbar(s_y, s_x, xerr=s_err, label=ntot, color='k', fmt='d', linestyle="None")  # ,

    # Plot stacked plot of components:
    _tab = tabel_dT_slcfs.loc[yr].transpose().rename({'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})

    a = _tab.plot(kind='barh', stacked=True, ax=ax, legend=(yr != '2040'))  # , grid=True)#stac)
    if not yr == '2040':
        ax.legend()  # [l],labels=['Sce!!nario total'], loc = 4)#'lower right')
    # Zero line:
    ax.axvline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_xlabel('$\Delta$ GMST ($^\circ$C)')
    ax.xaxis.set_minor_locator(MultipleLocator(.1))
    ax.grid(axis='y', which='major')

fn = RESULTS_DIR + '/figures/stack_bar_influence_years.png'
plt.tight_layout()
ax = plt.gca()

ax.tick_params(axis='y', which='minor')  # ,bottom='off')
ax.tick_params(labelright=True, right=True, left=False)
plt.savefig(fn, dpi=300)


# %% [markdown]
# ## Error bars from model uncertainty AND ECS uncertainty

# %% [markdown]
# See [Uncertainty_calculation.ipynb](Uncertainty_calculation.ipynb)

# %%
def sigma_DT(dT, sig_alpha, mu_alpha, dim='climatemodel'):
    sig_DT = dT.std(dim)
    mu_DT = dT.mean(dim)
    return ((sig_DT + mu_DT) * (sig_alpha + mu_alpha) - mu_DT * mu_alpha) / mu_alpha


def sigma_com(sig_DT, mu_DT, sig_alpha, mu_alpha, dim='climatemodel'):
    return (((sig_DT ** 2 + mu_DT ** 2) * (
            sig_alpha ** 2 + mu_alpha ** 2) - mu_DT ** 2 * mu_alpha ** 2) / mu_alpha ** 2) ** (.5)


sum_DT_std =  table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year, sts='std')
sum_DT_mean = table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year, sts='mean')
tot_DT_std = table_of_sts(ds_DT, scenarios_nhist, ['Delta T|Anthropogenic'], ['Total'], years, ref_year, sts='std')
tot_DT_mean = table_of_sts(ds_DT, scenarios_nhist, ['Delta T|Anthropogenic'], ['Total'], years, ref_year, sts='mean')

yerr_sum = sigma_com(sum_DT_std, sum_DT_mean, .24, .885)
yerr_tot = sigma_com(tot_DT_std, tot_DT_mean, .24, .885)  # .rename('')

# tab_sig_DT = setup_table_prop()

# %%
from matplotlib import transforms
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, len(years), figsize=[12, 6], sharex=False, sharey=True)
tits = ['Near Term surface temperature change (2040 relative to 2021)',
        'Long Term surface T change 2100 relatie to 2021)']
tits = ['Change in GMST in 2040 relative to 2021', 'Change in GMST in 2100 relative to 2021']
for yr, ax, tit in zip(years, axs, tits):
    ntot = 'Scenario total'
    tot_yr = tabel_dT_anthrop.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    tot_sd_yr = yerr_tot.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # l =ax.bar(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2, yerr=tab_tot_sd)
    sum_yr = tabel_dT_sum_slcf.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    sum_sd_yr = yerr_sum.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    ax.barh(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2,
            xerr=tot_sd_yr.transpose()[ntot].values,
            error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ntot = 'Sum SLCFs'
    # ax.bar(sum_yr.transpose().index, sum_yr.transpose()[ntot].values, color='r', label=ntot, alpha=.2, yerr=sum_sd_yr.transpose()[ntot].values,
    #      error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=1))

    s_x = sum_yr.transpose().index
    s_y = sum_yr.transpose()[ntot].values
    s_err = sum_sd_yr.transpose()[ntot].values
    ax.errorbar(s_y, s_x, xerr=s_err, label=ntot, color='k', fmt='d', linestyle="None")  # ,
    # error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=1))

    _tab = tabel_dT_slcfs.loc[yr].transpose().rename({'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})

    a = _tab.plot(kind='barh', stacked=True, ax=ax, legend=(yr != '2040'))  # , grid=True)#stac)
    if not yr == '2040':
        ax.legend()  # [l],labels=['Sce!!nario total'], loc = 4)#'lower right')

    ax.axvline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_xlabel('$\Delta$ GMST ($^\circ$C)')
    ax.xaxis.set_minor_locator(MultipleLocator(.1))
    ax.grid(axis='y', which='major')

fn = RESULTS_DIR + '/figures/stack_bar_influence_years_horiz_errTot.png'
plt.tight_layout()
ax = plt.gca()

ax.tick_params(axis='y', which='minor')  # ,bottom='off')
ax.tick_params(labelright=True, right=True, left=False)
plt.savefig(fn, dpi=300)

# %% [markdown]
# - De vi allerede har.
#
# Hvis vi skulle lagt til usikkerhet?
# - Ville lagt til usikkerhet gjennom ECS -- 
#     - Regne ut de samme tallene for 3 verdier av ECS. 
#     - Monte carlo trekk med en fordeling på ECS. 
#     - 
