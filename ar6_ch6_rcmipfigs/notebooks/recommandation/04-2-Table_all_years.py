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
import xarray as xr
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Paths input data

# %%
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

# PATH_DATASET = OUTPUT_DATA_DIR + '/forcing_data_rcmip_models.nc'
# PATH_DT = OUTPUT_DATA_DIR / '/dT_data_rcmip_models.nc'
PATH_DT = OUTPUT_DATA_DIR / 'dT_data_RCMIP_recommendation.nc'


# %% [markdown]
# #### Uncertainty data from Chris

# %%
PATH_DT_UNCERTAINTY = OUTPUT_DATA_DIR / 'dT_uncertainty_data_FaIR_chris.nc'


# %% [markdown]
# ## Set values:

# %%
first_y = 1750
last_y = 2100

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = 2020

# %%
FIGURE_DIR = RESULTS_DIR / 'figures_recommendation/'

TABLE_DIR = RESULTS_DIR / 'tables_recommendation/'

# %%
from pathlib import Path
Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)

# %%
percentile = 'percentile'
climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'
name_deltaT = 'Delta T'


# %% [markdown]
#
# ### Define variables to look at:

# %%
# variables to plot:
variables_erf_comp = [
    'ch4',
    # 'aerosol-radiation_interactions',
    # 'aerosol-cloud_interactions',
    'aerosol-total',
    'o3',
    'HFCs',
    'bc_on_snow']
# total ERFs for anthropogenic and total:
#variables_erf_tot = ['total_anthropogenic',
#                     'total']

scenarios_fl_370 = ['ssp370', 'ssp370-lowNTCF-aerchemmip', 'ssp370-lowNTCF-gidden'  # Due to mistake here
                    ]

# %% [markdown]
# ### Scenarios:

# %%
scenarios_fl = ['ssp119',
                'ssp126',
                'ssp245',
                'ssp370',
                'ssp370-lowNTCF-aerchemmip',
                #'ssp370-lowNTCF-gidden',
                'ssp370-lowNTCF-gidden',
                'ssp585']

# %%
median = 'median'
perc5 = '5th percentile'
perc95 = '95th percentile'
recommendation = 'recommendation'


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
ds_uncertainty = xr.open_dataset(PATH_DT_UNCERTAINTY)

# %%
for var in variables_erf_comp:
    da5 = ds_uncertainty.sel(variable=var, scenario='ssp585')['p05-p50']    
    da95 = ds_uncertainty.sel(variable=var, scenario='ssp585')['p95-p50']
    da5.plot(label=var)
    da95.plot(label=var)


# %% [markdown]
# ## Merge uncertainty and original: 
#

# %%
percentiles_to_keep = ['p05-p50','p16-p50','p84-p50','p95-p50']

# %%
_da = ds_uncertainty[percentiles_to_keep].to_array('percentile')
_ds = _da.rename(name_deltaT).to_dataset()

# %%
ds_DT= xr.concat([ds_DT[name_deltaT],_ds[name_deltaT]], dim='percentile').to_dataset()

# %%
ds_DT

# %% [markdown]
# ## Make csv table of total values:

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_var_nicename

# %%

# %%
reference_year = ref_year
start_y_tabel = 2015
end_y_tabel = last_y


# %%
def get_fn(var_name, s_y, e_y, ref_y, perc):
    _st = var_name.replace('(','').replace(')','').replace(' ','_').replace(',','')#+'.csv'
    fn = f'{perc}_{_st}_{s_y}-{e_y}_refyear{ref_y}.csv'
    return fn

def make_sum_slcfs_tabel(percentile=recommendation):
    _str = ''
    _vl = [get_var_nicename(var).split('(')[0].strip() for var in variables_erf_comp]
    for var in _vl:
        _str += f'{var}, '
    vn_sum = 'Sum SLCF (%s)' % _str[:-2]

    fn = get_fn(vn_sum, start_y_tabel, end_y_tabel, ref_year, percentile)



    # ref year value:
    ds_DT_sy = ds_DT[name_deltaT].sel(
        variable=variables_erf_comp,
        year=reference_year
    ).sum(variable).squeeze()

    # all values from s_y to e_y
    _da = ds_DT[name_deltaT].sel(
        variable=variables_erf_comp,
        year=slice(start_y_tabel, end_y_tabel)
    ).sum(variable).sel(year=slice(int(start_y_tabel), int(end_y_tabel))) - ds_DT_sy

    # Choose recommendation::
    _pl_da = _da.sel(percentile=percentile).squeeze()
    df = _pl_da.to_pandas().transpose()
    df['percentile'] = percentile
    display(df)
    fn = TABLE_DIR / fn
    df.to_csv(fn)
    df
for prc in [recommendation, 'p05-p50','p95-p50']:
    make_sum_slcfs_tabel(percentile=prc)
    print(prc)
# %%
import pandas as pd

# %%

# %%

# %% [markdown]
# ## TABLE EACH VAR

# %%

def make_slcfs_tabel(var, percentile=recommendation):
    _str = ''
    _st = get_fn(var, start_y_tabel, end_y_tabel, ref_year, percentile)


    ds_DT_sy = ds_DT[name_deltaT].sel(
        variable=var, 
        year=reference_year,
    ).squeeze()
    
    _da = ds_DT[name_deltaT].sel(
        variable=var,
        year=slice(start_y_tabel, end_y_tabel)) - ds_DT_sy
    
    # Take recommendation::
    _pl_da = _da.sel(percentile=recommendation).squeeze()

    df = _pl_da.to_pandas().transpose()
    df['percentile'] = percentile
    display(df)
    fn = TABLE_DIR/_st
    df.to_csv(fn)

for var in variables_erf_comp:
    for prc in [recommendation, 'p05-p50','p95-p50']:
        make_slcfs_tabel(var, percentile=prc)
