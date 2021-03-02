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

# %%

# %%
import pandas as pd
import xarray as xr
from IPython.display import clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR, OUTPUT_DATA_DIR, RESULTS_DIR

# %%

# %%
# name of output variable
name_deltaT = 'Delta T'

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'
percentile = 'percentile'

# %% [markdown]
# ### Define variables to look at:

# %%
# variables to include:

variables_erf_comp = [
    'ch4',
    'aerosol-total-with_bc-snow',
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
variables_in_sum = [
    'aerosol-total-with_bc-snow',
    'ch4',
    # 'aerosol-radiation_interactions',
    # 'aerosol-cloud_interactions',
    #'aerosol-total',
    'o3',
    'HFCs',
    #'bc_on_snow'
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

# %%
recommendation = 'recommendation'
IRFpercentiles = [recommendation]
# {'ECS = 2K':0.526, 'ECS = 3.4K':0.884, 'ECS = 5K': 1.136 }

# %% [markdown]
# Year to integrate from and to:

# %%
first_y = 1750
last_y = 2100

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = 2019

# %% [markdown]
# **Years to output change in**

# %%
years = [2040, 2100]

# %% [markdown]
# ### Input dataset:

# %%
PATH_DT_INPUT = OUTPUT_DATA_DIR / 'dT_data_RCMIP_recommendation.nc'


# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
PATH_DT_TAB_OUTPUT = RESULTS_DIR / 'tables_recommendation' / 'table_sens_dT_cs_recommandetion.csv'








# %%
print(PATH_DT_INPUT)

# %%
    
    
ds_DT = xr.open_dataset(PATH_DT_INPUT)

# %%
ds_DT.variable

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
                
                da_ref_y = ds_DT[dtvar].sel(
                    percentile=key,
                    scenario=scn, 
                    year=ref_year,
                ).squeeze()
                
                da_sel_y =ds_DT[dtvar].sel(
                    percentile=key,
                    scenario=scn, 
                    year=year
                ) 
                _tab_da = da_sel_y - da_ref_y
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
                _da_y = ds_DT[dtvar].sel(
                    percentile=key,
                    scenario=scn, 
                    year=year, 
                    variable=var
                ).squeeze()  # .squeeze()
                _da_refy = ds_DT[dtvar].sel(
                    scenario=scn, 
                    year=ref_year, 
                    variable=var,
                ).squeeze()
                # _tab_da = dic_ds[key][dtvar].sel(scenario=scn, time=slice(year,year))-  dic_ds[key][dtvar].sel(scenario=scn, time=slice(ref_year,ref_year)).squeeze()
                _tab_da = _da_y - _da_refy

                tab.loc[(scn, tabvar), (key, year)] = _tab_da.squeeze().values  # [0]

# %%
tab


# %%
tab

# %% [markdown]
# ## Save output

# %%
tab.to_csv(PATH_DT_TAB_OUTPUT)

# %%
PATH_DT_TAB_OUTPUT

# %%
