# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocesses uncertainty intervals from FaIR simulations
#
# This notebook reads in CSV files with percentiles for the temperature change in the scenarios as simulated with FaIR and then writes these to a netcdf file along.

# %%
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# %%

# %load_ext autoreload
# %autoreload 2

from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR_BADC

# %%
from ar6_ch6_rcmipfigs.utils.badc_csv import read_csv_badc

# %%
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR


# %% [markdown]
# ## Define file paths

# %% [markdown]
# ### Input:

# %%
fp_uncertainty = INPUT_DATA_DIR_BADC / 'slcf_warming_ranges/'

# %% [markdown]
# ### Output:

# %%
PATH_DT_OUTPUT = OUTPUT_DATA_DIR / 'fig6_22_and_6_24'/'dT_uncertainty_data_FaIR_chris.nc'

# %% [markdown] tags=[]
# # Code + figures

# %% [markdown]
# ### Read in csv files:

# %%


dic_ssp = {}
for f in fp_uncertainty.glob('*.csv'):
    ls_com = f.name.split('_')
    ssp = ls_com[-2]
    ssp
    perc = ls_com[-1].split('.')[0]
    if ssp not in dic_ssp.keys():
        dic_ssp[ssp] = dict()
    if perc not in dic_ssp[ssp]:
        dic_ssp[ssp][perc] = dict()

    dic_ssp[ssp][perc] = read_csv_badc(f, index_col=0)

# %% [markdown]
# ### Various definitions

# %%
percentiles = ['p05', 'p16', 'p50', 'p84', 'p95']

# %%
scenarios_fl = ['ssp119',
                'ssp126',
                'ssp245',
                'ssp370',
                'ssp370-lowNTCF-aerchemmip',
                'ssp370-lowNTCF-gidden',
                'ssp585']

# %%
dic_vars = dict(
    hfc='HFCs',
    o3='o3',
    slcf='Sum SLCF (Aerosols, Methane, Ozone, HFCs)',
    aerosol='aerosol-total',
    anthro='total_anthropogenic'
)
dic_cols = dict(
    forcing='variable',

)

# %% [markdown]
# ### Rename variables: 

# %%
for s in dic_ssp.keys():
    for p in dic_ssp[s].keys():
        _df = dic_ssp[s][p]
        _df = _df.rename(dic_vars, axis=1)
        dic_ssp[s][p] = _df

# %%
scenario_index = scenarios_fl + [scn for scn in dic_ssp.keys() if scn not in scenarios_fl]

# %% [markdown]
# ### Convert to dataset

# %%
dic_ssp_xr = dict()
ds = xr.Dataset()
for p in percentiles:
    dic_perc_xr = dict()
    for s in scenario_index:
        dic_ssp_xr[s] = dic_ssp[s][p].to_xarray()
    ls_s = [dic_ssp_xr[s] for s in scenario_index]
    _ds = xr.concat(ls_s, pd.Index(scenario_index, name='scenario'))
    _da = _ds.to_array(name=p)
    ds[p] = _da

# %% [markdown]
# ### Open delta T dataset to plot together

# %%
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR

PATH_DT = OUTPUT_DATA_DIR /'fig6_22_and_6_24'/'dT_data_RCMIP_recommendation.nc'

ds_DT = xr.open_dataset(PATH_DT)

# %%
ds

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_scenario_c_dic

# scenario colors and linestyle
cdic = get_scenario_c_dic()

# %% [markdown]
# ### Plot data:

# %%
cdic: dict

for var in ['o3', 'ch4', 'aerosol-total', 'bc_on_snow', 'HFCs']:
    for scn in scenarios_fl:
        pl_da = ds_DT['Delta T'].sel(variable=var, year=slice(2020, 2100), scenario=scn, percentile='recommendation') - \
                ds_DT['Delta T'].sel(variable=var, year=2020, scenario=scn, percentile='recommendation')

        pl_da.plot(linestyle='dashed', label=scn, c=cdic[scn])

        pl_da = ds['p50'].sel(variable=var, year=slice(2020, 2100), scenario=scn)  # , percentile='recommendation')
        pl_da.plot(linestyle='solid', label=scn, c=cdic[scn])
    plt.show()

# %% [markdown]
# ## Make difference between 50th percentile and others and save:

# %%
p50 = 'p50'
for perc in percentiles:
    nvn = f'{perc}-p50'
    ds[nvn] = ds[perc] - ds[p50]
    print(nvn)
ds

# %%
PATH_DT_OUTPUT

# %%
ds.to_netcdf(PATH_DT_OUTPUT)

# %%

# %%
