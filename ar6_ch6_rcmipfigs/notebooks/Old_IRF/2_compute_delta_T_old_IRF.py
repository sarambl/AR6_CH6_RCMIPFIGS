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
# TA MED LOW-NTF-GIDDENS!!!

# %% [markdown]
# # Plot temperature response over time

# %% [markdown]
# This notebook does the same as [2_compute_delta_T.ipynb](2_compute_delta_T.ipynb) except that it varies the ECS parameter and outputs a table of changes in temperature with respect to some reference year (defined below).

# %%
import pandas as pd

# %%
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR

# %% [markdown]
# **Output table found in:**
# %%


# %% [markdown]
# ### General about computing $\Delta T$:
# %% [markdown]
# We compute the change in GSAT temperature ($\Delta T$) from the effective radiative forcing (ERF) estimated from [Smith 2020](https://zenodo.org/record/3973015), by integrating with the impulse response function (IRF(t-t'))
# #todo: check for ref for IRF
# (Geoffroy at al 2013).
#
# For any forcing agent $x$, with estimated ERF$_x$, the change in temperature $\Delta T$ is calculated as:
#
# %% [markdown]
# \begin{align*}
# \Delta T_x (t) &= \int_0^t ERF_x(t') IRF(t-t') dt' \\
# \end{align*}
# %% [markdown]
# #### The Impulse response function (IRF):
# In these calculations we use:
# \begin{align*}
# IRF(t) = \frac{q_1}{d_1} \exp\Big(\frac{-t}{d_1}\Big) + \frac{q_2}{d_2} \exp\Big(\frac{-t}{d_2}\Big)
# \end{align*}
#
# Where the constants, $q_i$ and $d_i$ are from XXXXXX????
# %% [markdown]
# ## Input data:
# See [README.md](../../README.md)
# %% [markdown]
# # Code + figures

# %%
fn_IRF_constants = INPUT_DATA_DIR / 'irf_from_2xCO2_2020_12_02_050025-1.csv'
#fn_IRF_constants = INPUT_DATA_DIR / 'irf_from_2xCO2_2021_02_02_025721.csv'
irf_consts = pd.read_csv(fn_IRF_constants).set_index('id')

ld1 = 'd1 (yr)'
ld2 = 'd2 (yr)'
lq1 = 'q1 (K / (W / m^2))'
lq2 = 'q2 (K / (W / m^2))'
median = 'median'
perc5 = '5th percentile'
perc95 = '95th percentile'
irf_consts  # [d1]

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

PATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'

PATH_DT_TAB_OUTPUT = RESULTS_DIR / 'tables' / 'table_sens_dT_cs_old_IRF.csv'
PATH_DT_OUTPUT = OUTPUT_DATA_DIR / 'dT_data_RCMIP_old_IRF.nc'

# %% [markdown]
# **Output table found in:**

# %% pycharm={"name": "#%%\n"}
print(PATH_DT_OUTPUT)

# %% [markdown]
# ## Imports:

# %%
import xarray as xr
from IPython.display import clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

# %%

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'
percentile = 'percentile'

# %% [markdown]
# ## Set values:

# %% [markdown]
# ECS parameters:

# %%
IRFpercentiles = [perc5, median, perc95]
# {'ECS = 2K':0.526, 'ECS = 3.4K':0.884, 'ECS = 5K': 1.136 }

# %% [markdown]
# Year to integrate from and to:

# %%
first_y = '1750'
last_y = '2100'

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = '2021'

# %% [markdown]
# **Years to output change in**

# %%
years = ['2040', '2100']


# %% [markdown]
# ## IRF:

# %%

def IRF(t, d1, q1, d2, q2):
    """
    Returns the IRF function for:
    :param q2:
    :param d2:
    :param q1:
    :param d1:
    :param t: Time in years
    :return:
    IRF
    """
    irf = q1 / d1 * np.exp(-t / d1) + q2 / d2 * np.exp(-t / d2)
    return irf
    # l * (alpha1 * np.exp(-t / tau1) + alpha2 * np.exp(-t / tau2))


# %% [markdown]
# ## ERF:
# Read ERF from file

# %% [markdown]
# ### Define variables to look at:

# %%
# variables to plot:
variables_erf_comp = [
    'ch4',
    'aerosol-radiation_interactions',
    'aerosol-cloud_interactions',
    'aerosol-total',
    'o3',
    'HFCs',
    # 'F-Gases|HFC',
    'bc_on_snow',
    'total_anthropogenic',
    'total',
]
# total ERFs for anthropogenic and total:
variables_erf_tot = ['total_anthropogenic',
                     'total']
variables_all = variables_erf_comp + variables_erf_tot
# Scenarios to plot:
scenarios_fl = ['ssp534-over', 'ssp119', 'ssp460', 'ssp585', 'ssp370',
                'ssp370-lowNTCF-aerchemmip', 'ssp126', 'ssp245', 'ssp434',
                'ssp370-lowNTCF-gidden'
                ]

# %% [markdown]
# ### Open dataset:

# %%
ds = xr.open_dataset(PATH_DATASET).sel(year=slice(1700, 2200))  # we need only years until 1700
da_ERF = ds['ERF']

# ds['time'] = \
# ds['year'].to_pandas().index.map('{}-01-01'.format)
ds['time'] = pd.to_datetime(ds['year'].to_pandas().index.map(str), format='%Y')

# delta_t is 1 (year)
ds['delta_t'] = xr.DataArray(np.ones(len(ds['year'])), dims='year', coords={'year': ds['year']})

# %% [markdown]
# ## Integrate:
# The code below integrates the read in ERFs with the pre defined impulse response function (IRF).

# %% [markdown]
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %%
name_deltaT = 'Delta T'


def integrate_(i, _var, _nvar, ds_in: xr.Dataset, ds_DT, irf_cnst: dict):
    """

    :param i:
    :param _var:
    :param _nvar:
    :param ds_in:
    :param ds_DT:
    :param irf_cnst: dictionary
    :return:
    """
    # lets create a ds that goes from 0 to i inclusive
    ds_short = ds_in[{'year': slice(0, i + 1)}].copy()
    #print(ds_short)
    # lets get the current year
    current_year = ds_short['year'][{'year': i}]  # .dt.year
    # lets get a list of years
    _years = ds_short['year']  # .dt.year
    # lets get the year delta until current year(i)
    ds_short['end_year_delta'] = current_year - _years

    # lets get the irf values from 0 until i
    d1 = irf_cnst[ld1]
    d2 = irf_cnst[ld2]
    q1 = irf_cnst[lq1]
    q2 = irf_cnst[lq2]

    ds_short['irf'] = IRF(
        ds_short['end_year_delta'] * ds_short['delta_t'], d1, q1, d2, q2)

    # lets do the famous integral
    ds_short['to_integrate'] = \
        ds_short[_var] * \
        ds_short['irf'] * \
        ds_short['delta_t']

    # lets sum all the values up until i and set
    # this value at ds_DT
    # If whole array is null, set value to nan
    if np.all(ds_short['to_integrate'].isnull()):  # or last_null:
        _val = np.nan
    else:
        # 

        _ds_int = ds_short['to_integrate'].sum(['year'])
        # mask where last value is null (in order to not get intgral 
        _ds_m1 = ds_short['to_integrate'].isel(year=-1)
        # where no forcing data)
        _val = _ds_int.where(_ds_m1.notnull())
    # set value in dataframe:
    ds_DT[_nvar][{'year': i}] = _val


def integrate_to_dT(_ds, from_t, to_t, irf_cnsts, int_var='ERF'):
    """
    Integrate forcing to temperature change.

    :param _ds: dataset containing the forcings
    :param from_t: start year
    :param to_t: end year
    :param int_var: variables to integrate
    :param irf_cnsts: irf constants
    :return:
    """
    # slice dataset
    ds_sl = _ds.sel(year=slice(from_t, to_t))
    len_time = len(ds_sl['year'])
    # lets create a result DS
    ds_DT = ds_sl.copy()

    # lets define the vars of the ds
    namevar = name_deltaT
    # set all values to zero for results dataarray:
    ds_DT[namevar] = ds_DT[int_var] * 0
    # Units Kelvin:
    ds_DT[namevar].attrs['unit'] = 'K'
    if 'unit' in ds_DT[namevar].coords:
        ds_DT[namevar].coords['unit'] = 'K'

    for i in range(len_time):
        # da = ds[var]
        if (i % 20) == 0:
            print('%s of %s done' % (i, len_time))
        integrate_(i, int_var, namevar, ds_sl, ds_DT, irf_cnsts)
    clear_output()
    # fn = 'DT_%s-%s.nc' % (from_t, to_t)
    # fname = OUTPUT_DATA_DIR/ fn#'DT_%s-%s.nc' % (from_t, to_t)
    # save dataset.
    # ds_DT.to_netcdf(fname)
    return ds_DT


# %% [markdown]
# ## Compute $\Delta T$ with 3 different climate sensitivities

# %%
ds

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
# noinspection PyStatementEffect
irf_consts.loc[median][ld1]

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
dic_ds = {}
for key in IRFpercentiles:
    dic_ds[key] = integrate_to_dT(ds, first_y, last_y, irf_consts.loc[key], int_var='ERF')

# %% [markdown]
# ## check:

# %%
for per in IRFpercentiles:
    dic_ds[per].isel(scenario=0, variable=0)[name_deltaT].plot()

# %% [markdown]
# ### Make datset with percentile as dimension:

# %%
ds_tmp = xr.Dataset(coords=dic_ds[median].coords)
ds_tmp
for key in IRFpercentiles:
    ds_tmp[key] = dic_ds[key]['Delta T']  # .dims,dic_ds[key],)
ds['Delta T'] = ds_tmp.to_array('percentile')
ds.sel(year=slice(first_y, last_y)).to_netcdf(PATH_DT_OUTPUT)
# ds_DT.to_array('percentile')
# dic_ds[key]['Delta T']

# %% [markdown]
# ## Table

# %% [markdown]
# ### Setup table:

# %%

iterables = [list(IRFpercentiles), years]


def setup_table(scenario_n='', variables=variables_all):
    _i = pd.MultiIndex.from_product(iterables, names=['', ''])
    table = pd.DataFrame(columns= variables, index=_i).transpose()
    table.index.name = scenario_n
    return table


# %%


# %%
# Dicitonary of tables with different ESC:
scntab_dic = {}
for scn in scenarios_fl:
    # Loop over scenrarios
    tab = setup_table(scenario_n=scn, variables=variables_erf_comp)  # make table
    for var in variables_erf_comp:
        # Loop over variables
        tabvar = var.split('|')[-1]
        dtvar = name_deltaT
        for key in IRFpercentiles:
            # Loop over ESC parameters
            for year in years:
                _tab_da = dic_ds[key][dtvar].sel(scenario=scn, year=slice(year, year)) - dic_ds[key][dtvar].sel(
                    scenario=scn, year=slice(ref_year, ref_year)).squeeze()
                a = float(_tab_da.loc[var].squeeze().values)
                tab.loc[tabvar, (key, year)] = a
    scntab_dic[scn] = tab.copy()

# %%
from IPython.display import display

for key in scntab_dic:
    display(scntab_dic[key])

# %% [markdown]
# ### Make table with all scenarios:

# %%
iterables = [list(IRFpercentiles), years]
iterables2 = [scenarios_fl, variables_erf_comp]


def setup_table2():  # scenario_n=''):
    _i = pd.MultiIndex.from_product(iterables, names=['', ''])
    _r = pd.MultiIndex.from_product(iterables2, names=['', ''])

    table = pd.DataFrame(columns=_r, index=_i).transpose()
    return table


# %%
tab = setup_table2()  # scenario_n=scn)

for scn in scenarios_fl:
    for var in variables_erf_comp:
        tabvar = var#.split('|')[-1]
        dtvar = name_deltaT
        for key in IRFpercentiles:
            for year in years:
                # compute difference between year and ref year
                _da_y = dic_ds[key][dtvar].sel(scenario=scn, year=slice(year, year), variable=var)  # .squeeze()
                _da_refy = dic_ds[key][dtvar].sel(scenario=scn, year=slice(ref_year, ref_year), variable=var).squeeze()
                # _tab_da = dic_ds[key][dtvar].sel(scenario=scn, time=slice(year,year))-  dic_ds[key][dtvar].sel(scenario=scn, time=slice(ref_year,ref_year)).squeeze()
                _tab_da = _da_y - _da_refy

                tab.loc[(scn, tabvar), (key, year)] = _tab_da.squeeze().values  # [0]

# %%
tab

# %% [markdown]
# ## Save output

# %%
tab.to_csv(PATH_DT_TAB_OUTPUT)

# %%
PATH_DT_TAB_OUTPUT

# %% [markdown]
# ## Double check historical $\Delta$ T: 
#

# %%
from matplotlib.ticker import (MultipleLocator)

from ar6_ch6_rcmipfigs.utils.plot import get_cmap_dic

# %%
ls_vars = ['aerosol-total', 'ch4', 'co2', 'other_wmghg', 'o3','HFCs']

# %%
cdic = get_cmap_dic(ls_vars)

# %%
ds

# %%

fig, ax = plt.subplots(figsize=[6, 7])

ds_hist = ds.sel(year=slice(1750, 2019), percentile='median', scenario='ssp119', variable=ls_vars)
for var in ds_hist.variable.values:
    ds_hist.sel(variable=var)[name_deltaT].plot(label=var, linestyle='dashed', linewidth=2, c=cdic[var])
plt.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')  # , prop=fontP)
ax.yaxis.set_major_locator(MultipleLocator(.25))
ax.yaxis.set_minor_locator(MultipleLocator(.05))
ax.tick_params(right=True, labelright=True)
ax.set_xlim([1750, 2019])
ax.set_ylim([-0.9, 1.25])
plt.title('')
plt.savefig('test.pdf')
plt.show()

# %%

# %%
da_ERF.scenario

# %%
