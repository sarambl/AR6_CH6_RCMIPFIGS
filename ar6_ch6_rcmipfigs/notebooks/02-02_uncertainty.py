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
import matplotlib.pyplot as plt
import pandas as pd

# %%
from ar6_ch6_rcmipfigs import constants

# %load_ext autoreload
# %autoreload 2

from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR

# %%
first_y = '1750'
last_y = '2100'

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = '2020'

# %% [markdown]
# # Code + figures

# %%
fn_uncertainty = INPUT_DATA_DIR /'chris_slcf_warming_ranges.csv'
df_uncertainty = pd.read_csv(fn_uncertainty, index_col=0)#.set_index('id')

# make sure base period/ref period are the same:
df_uncertainty = df_uncertainty[df_uncertainty['base_period']==int(ref_year)]
df_uncertainty#['scenario']#.uniqu
#diff_uncertainty = df_uncertainty - df_uncertainty['p50']

# %% [markdown]
# ## Renaming to fit conventions:

# %% [markdown]
# #### variables:
#

# %% [markdown]
#     'ch4',
#     'aerosol-total',
#     'o3',
#     'HFCs',
#     'bc_on_snow']

# %%
dic_vars = dict(
    hfc='HFCs', 
    slcf='Sum SLCF (Aerosols, Methane, Ozone, HFCs)',
    aerosol='aerosol-total',
    anthro='total_anthropogenic'
)
dic_cols = dict(
    forcing='variable',
    
)


# %%
df_uncertainty['forcing'] = df_uncertainty['forcing'].replace(dic_vars)#.unique()
df_uncertainty = df_uncertainty.rename(dic_cols, axis=1)


# %%
percentiles =['p05','p16','p50','p84','p95']

# %%
ds = df_uncertainty.set_index(['scenario','variable','year', 'base_period']).to_xarray()

ds

# %%
ds.variable

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

PATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'

PATH_DT_TAB_OUTPUT = RESULTS_DIR / 'tables' / 'table_sens_dT_cs_recommandetion.csv'
PATH_DT_OUTPUT = OUTPUT_DATA_DIR / 'dT_uncertainty_data_FaIR_chris.nc'

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

# %% [markdown]
# Year to integrate from and to:

# %%
first_y = '1750'
last_y = '2100'

# %% [markdown]
# **Set reference year for temperature change:**

# %% [markdown]
# **Years to output change in**

# %%
years = ['2040', '2100']

# %%
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

# PATH_DATASET = OUTPUT_DATA_DIR + '/forcing_data_rcmip_models.nc'
# PATH_DT = OUTPUT_DATA_DIR / '/dT_data_rcmip_models.nc'
PATH_DT = OUTPUT_DATA_DIR / 'dT_data_RCMIP_recommendation.nc'


# %%
ds_DT = xr.open_dataset(PATH_DT)

# %%
ds_DT#.scenario  # .climatemodel

# %%
ds

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_scenario_c_dic, \
    get_scenario_ls_dic

# scenario colors and linestyle
cdic = get_scenario_c_dic()
lsdic = get_scenario_ls_dic()  # get_ls_dic(ds_DT[climatemodel].values)


# %%
scenario
scenarios_fl = ['ssp119',
                'ssp126',
                'ssp245',
                'ssp370',
                'ssp370-lowNTCF-aerchemmip',
                'ssp370-lowNTCF-gidden',
                'ssp585']

# %%

for var in ['o3','ch4','aerosol-total','bc_on_snow','HFCs']:
    for scn in scenarios_fl:
        pl_da = ds_DT['Delta T'].sel(variable=var, year=slice(2020,2100), scenario=scn, percentile='recommendation')-ds_DT['Delta T'].sel(variable=var, year=2020, scenario=scn, percentile='recommendation')

        pl_da.plot(linestyle = 'dashed', label=scn, c=cdic[scn])
        
        pl_da = ds['p50'].sel(variable=var, year=slice(2020,2100), scenario=scn)#, percentile='recommendation')
        pl_da.plot(linestyle = 'solid', label=scn, c=cdic[scn])
    plt.show()

# %% [markdown]
# ## Make difference and save:

# %%
p50= 'p50'
for perc in percentiles:
    nvn = f'{perc}-p50'
    ds[nvn] = ds[perc] -ds[p50]
    print(nvn)
ds

# %%
PATH_DT_OUTPUT

# %%
ds.to_netcdf(PATH_DT_OUTPUT)

# %% [markdown]
# ## Comparison with old version

# %%

# %%
fn_uncertainty_o = INPUT_DATA_DIR /'slcf_warming_ranges_old.csv'
df_uncertainty_o = pd.read_csv(fn_uncertainty_o, index_col=0)#.set_index('id')

# make sure base period/ref period are the same:
df_uncertainty_o = df_uncertainty_o[df_uncertainty_o['base_period']==int(ref_year)]
df_uncertainty_o#['scenario']#.uniqu
#diff_uncertainty = df_uncertainty - df_uncertainty['p50']

# %%
var = 'ch4'
perc = 'p05'

cdic = get_scenario_c_dic()
for scn in cdic.keys():
    pl_n= df_uncertainty#
    pl_n = pl_n[pl_n['variable']==var]
    pl_n = pl_n[pl_n['scenario']==scn]
    plt.plot(pl_n['year'], pl_n[perc], c=cdic[scn], alpha=0.7)
    pl_n= df_uncertainty_o#
    pl_n = pl_n[pl_n['forcing']==var]
    pl_n = pl_n[pl_n['scenario']==scn]
    plt.plot(pl_n['year'], pl_n[perc], c=cdic[scn], linestyle='dotted')
plt.show()

# %%
var = 'aerosol'
perc = 'p05'

cdic = get_scenario_c_dic()
for scn in cdic.keys():
    pl_n= df_uncertainty#
    pl_n = pl_n[pl_n['variable']==var]
    pl_n = pl_n[pl_n['scenario']==scn]
    plt.plot(pl_n['year'], pl_n[perc], c=cdic[scn], alpha=0.7)
    pl_n= df_uncertainty_o#
    pl_n = pl_n[pl_n['forcing']==var]
    pl_n = pl_n[pl_n['scenario']==scn]
    plt.plot(pl_n['year'], pl_n[perc], c=cdic[scn], linestyle='dotted')
plt.show()

# %%
var = 'aerosol-total'
var2 = 'aerosol'
perc = 'p95'

cdic = get_scenario_c_dic()
for scn in cdic.keys():
    pl_n= df_uncertainty#
    pl_n = pl_n[pl_n['variable']==var]
    pl_n = pl_n[pl_n['scenario']==scn]
    plt.plot(pl_n['year'], pl_n[perc], c=cdic[scn], alpha=0.7)
    pl_n= df_uncertainty_o#
    pl_n = pl_n[pl_n['forcing']==var2]
    pl_n = pl_n[pl_n['scenario']==scn]
    plt.plot(pl_n['year'], pl_n[perc], c=cdic[scn], linestyle='dotted')
plt.show()

# %%

# %%

# %%
