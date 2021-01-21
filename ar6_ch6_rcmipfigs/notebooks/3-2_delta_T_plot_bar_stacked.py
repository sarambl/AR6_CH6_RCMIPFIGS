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
# This notebook plots temperature respons to SLCFs AND the total scenario forcing in a fixed nr of years

# %% [markdown]
# ## Imports:

# %%
import xarray as xr
from IPython.display import clear_output
import numpy as np
import os
import re
from pathlib import Path
import pandas as pd
import tqdm
from scmdata import df_append, ScmDataFrame
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# %load_ext autoreload
# %autoreload 2

# %%
from ar6_ch6_rcmipfigs.constants import BASE_DIR
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, INPUT_DATA_DIR, RESULTS_DIR

#PATH_DATASET = OUTPUT_DATA_DIR + '/forcing_data_rcmip_models.nc'
#PATH_DT = OUTPUT_DATA_DIR / '/dT_data_rcmip_models.nc'
PATH_DT = OUTPUT_DATA_DIR / 'dT_data_RCMIP.nc'

# %% [markdown]
# ## Set values:

# %%
first_y ='1850'
last_y = '2100'
# Years to plot:
years = ['2040', '2100']

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = '2021'

# %%
FIGURE_DIR = RESULTS_DIR / 'figures/'

# %%

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'

# %%

name_deltaT = 'Delta T'

# %% [markdown]
# ### Define variables to look at:

# %%
# variables to plot:
variables_erf_comp = [
    'ch4',
    'aerosol-total',
    'o3',
    #'F-Gases|HFC',
    'bc_on_snow']
# total ERFs for anthropogenic and total:
variables_erf_tot = ['total_anthropogenic',
                     'total']
variables_all = variables_erf_comp + variables_erf_tot
# Scenarios to plot:
scenarios_fl = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF-aerchemmip',  # 'ssp370-lowNTCF', Due to mistake here
                'ssp585']#, 'historical']

# %%

scenarios_fl = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF-aerchemmip',  # 'ssp370-lowNTCF', Due to mistake here
                'ssp585', 'historical']

scenarios_fl_370 = ['ssp370', 'ssp370-lowNTCF-aerchemmip','ssp370-lowNTCF-gidden'# Due to mistake here
                ]


# %% [markdown]
# ### Scenarios:

# %%
scenarios_fl=['ssp119',
              'ssp126',
              'ssp245',
              'ssp370',
              'ssp370-lowNTCF-aerchemmip',
              #'ssp370-lowNTCF-gidden',
              'ssp585']

# %%
median = 'median'
perc5 = '5th percentile'
perc95 = '95th percentile'

# %%

variables_tot = ['Total']
variables_sum = ['Sum SLCFs']

def setup_table_prop(scenario_n='', years=None, variabs=None, scens=None):
    if variabs is None:
        variabs = variables_erf_comp
    if years is None:
        years = ['2040', '2100']
    if scens is None:
        scens = scenarios_fl
    its = [years, variabs]
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
from ar6_ch6_rcmipfigs.utils.plot import get_var_nicename

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_scenario_c_dic, get_scenario_ls_dic

# %%

s_y = first_y

cdic = get_scenario_c_dic()  # get_cmap_dic(ds_DT[scenario].values)
lsdic = get_scenario_ls_dic()  # _scget_ls_dic(ds_DT[climatemodel].values)

def sum_name(var):
    """
    Returns the name off the sum o
    """
    return f'{var} sum '#|'.join(var.split('|')[0:2]) + '|' + 'All'



# make xarray with variable as new dimension:
_lst_f = []
_lst_dt = []
# Make list of dataArrays to be concatinated:
for var in variables_erf_comp:
    _lst_f.append(ds_DT['ERF'].sel(variable = var))
    _lst_dt.append(ds_DT[name_deltaT].sel(variable=var))
# Name of new var:
erf_all = sum_name('ERF')
# Name of new var:
dt_all = sum_name( name_deltaT)
#ds_DT[erf_all] = xr.concat(_lst_f, pd.Index(variables_erf_comp, name='variable'))
#ds_DT[dt_all] = xr.concat(_lst_dt, pd.Index(variables_erf_comp, name='variable'))
dt_totn = dt_all

# %% [markdown]
# ### compute sum: 

# %%
sum_name = 'Sum SLCFs'
ds_sub = ds_DT.sel(variable=variables_erf_comp)
ds_sum = ds_sub.sum(variable)
ds_sum = ds_sum.assign_coords(coords={variable:sum_name})
# add sum to variable coordinate

ds_sum = xr.concat([ds_sum,ds_DT.sel(variable=variables_erf_comp)], dim=variable )

# %%
ds_sum#.assign_coords(coords={variable:sum_name})


# %%

# %%
scntab_dic = {}


# tab_tot = setup_table2()
# tab_tot_sd = setup_table2()
def table_of_sts(ds_DT, scenarios_fl, variables, tab_vars, years, ref_year, sts='median'):
    """
    Creates pandas dataframe of statistics (mean, median, standard deviation) for change
    in temperature Delta T since year (ref year) for each scenario in scenarios,

    :param ds_DT:
    :param scenarios_fl:
    :param variables:
    :param tab_vars:
    :param years:
    :param ref_year:
    :param sts:
    :return:
    """
    tabel = setup_table_prop(years=years, variabs=tab_vars)
    for scn in scenarios_fl:
        for var, tabvar in zip(variables, tab_vars):
            #dtvar =  name_deltaT # if ERF name, changes it here.
            tabscn = scn  # Table scenario name the same.
            for year in years:
                _da =ds_DT[name_deltaT].sel(scenario=scn, variable=var)
                _da_refy = _da.sel(year=slice(ref_year, ref_year)).squeeze() # ref year value
                _da_y = _da.sel(year=slice(year, year)) # year value
                _tab_da = _da_y - _da_refy
                #_tab_da = ds_DT[dtvar].sel(scenario=scn, time=slice(year, year)) - ds_DT[dtvar].sel(scenario=scn,
                #                                                                                    time=slice(ref_year,
                #                                                                                               ref_year)).squeeze()
                tabel.loc[(year, tabvar), tabscn] = float(_tab_da.sel(percentile=sts).squeeze().values )#[0]

    return tabel

def table_of_stats_varsums(ds_DT, scenarios, dsvar, tabvar, years, ref_year, sts='median'):
    """
    Sums up over dimension 'variable' and creates pandas dataframe of statistics (mean, median, standard deviation) for change
    in temperature Delta T since year (ref year) for each scenario in scenarios. 

    :param ds_DT:
    :param scenarios_fl:
    :param variables:
    :param tab_vars:
    :param years:
    :param ref_year:
    :param sts:
    :return:
    """
    tabel = setup_table_prop(years=years, variabs=[tabvar])
    da = ds_DT[name_deltaT].sel(variable = dsvar)

    for scn in scenarios_fl:
        #for var, tabvar in zip(variables, tab_vars):
        dtvar =  name_deltaT # if ERF name, changes it here.
        tabscn = scn  # Table scenario name the same.
        for year in years:
            _da =da.sel(scenario=scn)#, variable = dsvar)
            _da_refy = _da.sel(year=slice(ref_year, ref_year)).squeeze() # ref year value
            _da_y = _da.sel(year=slice(year, year)).squeeze() # year value
            _tab_da = (_da_y - _da_refy).squeeze()
            
            #if sts=='mean':
            #    _tab_da = _tab_da.mean('climatemodel').sum('variable')
            #elif sts == 'median':
            #    _tab_da = _tab_da.median('climatemodel').sum('variable')
            #elif sts=='std':
            #    _tab_da= _tab_da.sum('variable').std('climatemodel')

            # Do statistics over RCMIP models
            tabel.loc[(year, tabvar), tabscn] = float(_tab_da.sel(percentile=sts).squeeze().values )#[0]
            float(_tab_da.sel(percentile=sts).squeeze().values )
            #tabel.loc[(year, tabvar), tabscn] = _tab_da.values

    return tabel


# %% [markdown]
# ### Computes statistics:

# %%
ds_DT.percentile

# %%
ds_DT.variable.values

# %%
# Statistics on Delta T anthropogenic
# Mean
tabel_dT_anthrop = table_of_sts(ds_DT, scenarios_fl, ['total_anthropogenic'], ['Total'], years, ref_year)

# Standard deviation
tabel_dT_anthrop_5th = table_of_sts(ds_DT, scenarios_fl, ['total_anthropogenic'], ['Total'], years, ref_year, sts='5th percentile')
tabel_dT_anthrop_95th = table_of_sts(ds_DT, scenarios_fl, ['total_anthropogenic'], ['Total'], years, ref_year, sts='95th percentile')

# %%
# Mean:
tabel_dT_slcfs = table_of_sts(ds_DT, scenarios_fl, variables_erf_comp, variables_erf_comp, years,
                              ref_year)
# Standard deviation
tabel_dT_slcfs_5th = table_of_sts(ds_DT, scenarios_fl, variables_erf_comp, variables_erf_comp,
                                 years, ref_year, sts='5th percentile')
tabel_dT_slcfs_95th = table_of_sts(ds_DT, scenarios_fl, variables_erf_comp, variables_erf_comp,
                                 years, ref_year, sts='5th percentile')

# Compute sum of SLCFs
_ds = ds_sum.copy()
vall = 'Delta T'
#_ds['test'] = _ds.sel(variable='Sum SLCFs')


#tabel_dT_sum_slcf = table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year)
#tabel_dT_sum_slcf_SD = table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year, sts='std')

# %%
tabel_dT_sum_slcf = table_of_stats_varsums(ds_sum, scenarios_fl, 'Sum SLCFs', 'Sum SLCFs', years, ref_year)
tabel_dT_sum_slcf_5 = table_of_stats_varsums(ds_sum, scenarios_fl, 'Sum SLCFs', 'Sum SLCFs', years, ref_year, sts='5th percentile')
tabel_dT_sum_slcf_95 = table_of_stats_varsums(ds_sum, scenarios_fl, 'Sum SLCFs', 'Sum SLCFs', years, ref_year, sts='95th percentile')
tabel_dT_sum_slcf

# %%
rn_dic = {}
for v in variables_all:
    rn_dic[v]= get_var_nicename(v)

# %%
tabel_dT_slcfs = tabel_dT_slcfs.rename(rn_dic)

# %% [markdown]
# ## Error bars only from model uncertainty
# The following uncertainties assume the ECS has a standard deviation of

# %%
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, len(years), figsize=[10, 4], sharex=False, sharey=True)
tits = ['Near Term surface temperature change (2040 relative to 2021)',
        'Long Term surface T change 2100 relatie to 2021)']
tits = ['Change in GSAT in 2040 relative to 2021', 'Change in GSAT in 2100 relative to 2021']
for yr, ax, tit in zip(years, axs, tits):
    ntot = 'Scenario total'
    # Pick out year and do various renames:
    # Total antropogenic
    tot_yr = tabel_dT_anthrop.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # Sum SLCFs
    sum_yr = tabel_dT_sum_slcf.loc[yr].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # Plot bars for anthropopogenic total:
    ax.barh(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2,
           )
    # Plot bars for SLCFs total:
    ntot = 'Sum SLCFs'
    s_x = sum_yr.transpose().index
    s_y = sum_yr.transpose()[ntot].values

    # Plot stacked plot of components:
    _tab = tabel_dT_slcfs.loc[yr].transpose().rename({'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})

    a = _tab.plot(kind='barh', stacked=True, ax=ax, legend=(yr != '2040'))  # , grid=True)#stac)
    if not yr == '2040':
        ax.legend(frameon=False)  # [l],labels=['Sce!!nario total'], loc = 4)#'lower right')
    # Zero line:
    ax.axvline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_xlabel('$\Delta$ GSAT ($^\circ$C)')
    ax.xaxis.set_minor_locator(MultipleLocator(.1))
    #ax.grid(axis='y', which='major')

ax = axs[0]

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params( right=False, left=False)#, color='w')

ax = axs[1]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(labelleft=False, right=False, left=False, color='w')
ax.tick_params(labelright=False, labelleft=False, right=False, left=False,  color='w')
ax.yaxis.set_visible(False)
plt.tight_layout()

fn = RESULTS_DIR / 'figures/stack_bar_influence_years.png'
plt.tight_layout()
ax = plt.gca()

plt.savefig(fn, dpi=300)

# %% [markdown]
# ## Error bars from model uncertainty AND ECS uncertainty

# %% [markdown]
# See [Uncertainty_calculation.ipynb](Uncertainty_calculation.ipynb)

# %% [markdown]
# ## Only ssp370:

# %%
scenario_370 =[sc for sc in scenarios_fl if 'ssp370' in sc]

# %%
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, len(years), figsize=[10, 2.5], sharex=False, sharey=True)

tits = ['Near Term surface temperature change (2040 relative to 2021)',
        'Long Term surface T change 2100 relatie to 2021)']
tits = ['Change in GSAT in 2040 relative to 2021', 'Change in GSAT in 2100 relative to 2021']
for yr, ax, tit in zip(years, axs, tits):
    ntot = 'Scenario total'
    # Pick out year and do various renames:
    # Total antropogenic
    tot_yr = tabel_dT_anthrop.loc[yr, scenario_370].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # Sum SLCFs
    sum_yr = tabel_dT_sum_slcf.loc[yr, scenario_370].rename({'Total': ntot, 'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    # Plot bars for anthropopogenic total:
    ax.barh(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2,
           )
    # Plot bars for SLCFs total:
    ntot = 'Sum SLCFs'
    s_x = sum_yr.transpose().index
    s_y = sum_yr.transpose()[ntot].values

    # Plot stacked plot of components:
    _tab = tabel_dT_slcfs.loc[yr, scenario_370].transpose().rename({'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})

    a = _tab.plot(kind='barh', stacked=True, ax=ax, legend=(yr != '2040'))  # , grid=True)#stac)
    if not yr == '2040':
        ax.legend(frameon=True)  # [l],labels=['Sce!!nario total'], loc = 4)#'lower right')
    # Zero line:
    ax.axvline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_xlabel('$\Delta$ GSAT ($^\circ$C)')
    ax.xaxis.set_minor_locator(MultipleLocator(.1))
    #ax.grid(axis='y', which='major')

ax = axs[0]

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params( right=False, left=False)#, color='w')

ax = axs[1]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(labelleft=False, right=False, left=False, color='w')
ax.tick_params(labelright=False, labelleft=False, right=False, left=False,  color='w')
ax.yaxis.set_visible(False)
plt.tight_layout()
fn = RESULTS_DIR / 'figures/stack_bar_influence_years_horiz_errTot_370only.png'

plt.tight_layout()
ax = plt.gca()

plt.savefig(fn, dpi=300)
