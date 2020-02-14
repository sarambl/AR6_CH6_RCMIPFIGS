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
# # Convert effective radiative forcings from RCMIP model output (csv/xlsx) to xarray Dataset:

# %% [markdown]
# This notebooks imports the data from the generated database and then selects the subset of variables needed and converts them into an  [xarray](http://xarray.pydata.org/en/stable/) Dataframe before saving as a [netCDF](https://www.unidata.ucar.edu/software/netcdf/) file. 

# %% [markdown]
# This step is purely to make pick out a subset of the data and make further computations simpler. 

# %% [markdown]
# ## Imports:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import xarray as xr
from IPython.display import clear_output
import numpy as np
import os
import re
from pathlib import Path
import pandas as pd
import tqdm
from scmdata import df_append, ScmDataFrame


# %load_ext autoreload
# %autoreload 2


# %%
from ar6_ch6_rcmipfigs.constants import BASE_DIR
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, INPUT_DATA_DIR

SAVEPATH_DATASET = OUTPUT_DATA_DIR + '/forcing_data_rcmip_models.nc'

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'

# %% [markdown]
# ### Define variables to look at:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# variables to load:
variables_erf = [
    'Effective Radiative Forcing|Anthropogenic|*',
    'Effective Radiative Forcing|Anthropogenic',
    'Effective Radiative Forcing',
    'Effective Radiative Forcing|Anthropogenic|Albedo Change|Other|Deposition of Black Carbon on Snow'
]
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
                'ssp370-lowNTCF-gidden',
                # 'ssp370-lowNTCF', Due to mistake here
                'ssp585', 'historical']

# %% [markdown]
# ### Models to look for

# %% [markdown]
# Models are chosen solely on availability of relevant  data:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

model_of_interest = [
    #    ".*acc2.*v2-0-1.*",
    ".*rcmip-phase-1_cicero-scm.*",
    #    ".*escimo.*v2-0-1.*",
    ".*fair-1.5-default.*",
    #    ".*rcmip_phase-1_gir.*",
    #    ".*greb.*v2-0-0.*",
    #    ".*hector.*v2-0-0.*",
    #    ".*MAGICC7.1.0aX-rcmip-phase-1.*",
    ".*rcmip-phase-1_magicc7.1.0.beta*",
    #    ".*MAGICC7.1.0aX.*",
    #    ".*mce.*v2-0-1.*",
    #    ".*oscar-v3-0*v1-0-1.*",
    ".*oscarv3.0.*"
    #    ".*wasp.*v1-0-1.*",
]
# %% [markdown]
# Where to look for files:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

# RESULTS_PATH = os.path.join(BASE_DIR, "data", "results", "phase-1/")
RESULTS_PATH = os.path.join(INPUT_DATA_DIR, "database-results", "phase-1")
RESULTS_PATH
# RESULTS_PATH

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
SCENARIO_PROTOCOL = os.path.join(INPUT_DATA_DIR, "data", "protocol", "rcmip-emissions-annual-means-v3-1-0.csv")
SCENARIO_PROTOCOL

# %% [markdown]
# List of files:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
results_files = list(Path(RESULTS_PATH).rglob("*.csv")) + list(Path(RESULTS_PATH).rglob("*.xlsx"))
results_files[:4]

# %% [markdown]
# ## Make file list to load:

# %%
results_files = [
    str(p)
    for p in results_files
    if any([bool(re.match(m, str(p))) for m in model_of_interest]) and "$" not in str(p)
]
print(len(results_files))
print(results_files)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from ar6_ch6_rcmipfigs.utils.misc_func import prep_str_for_filename

variables_of_interest = variables_erf + variables_erf_comp + variables_erf_tot
relevant_files = [
    str(p)
    for p in results_files
    if any(
        [
            bool(re.match(".*{}.*".format(prep_str_for_filename(v)), str(p)))
            for v in variables_of_interest
        ]
    )
]
print("Number of relevant files: {}".format(len(relevant_files)))
relevant_files

# %% [markdown]
# ### Remove quantile files:

# %%
quantile='quantile'
relevant_files= [
    str(p)
    for p in relevant_files
    if quantile not in p]
print("Number of relevant files: {}".format(len(relevant_files)))
relevant_files

# %% [markdown]
# ### Read in all variables:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
db = []
for rf in tqdm.tqdm_notebook(relevant_files):
    # print(rf.endswith('sf'))
    if rf.endswith(".csv"):
        loaded = ScmDataFrame(rf)
    else:
        loaded = ScmDataFrame(rf, sheet_name="your_data")
    db.append(loaded.filter(variable=variables_erf, scenario=scenarios_fl))  # variables_of_interest))
print(db)
db = df_append(db).timeseries().reset_index()
db["unit"] = db["unit"].apply(
    lambda x: x.replace("Dimensionless", "dimensionless") if isinstance(x, str) else x
)
clear_output()
db = ScmDataFrame(db)
db.head()

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
db[variable].unique()

# %%
db[climatemodel].unique()

# %%
db[scenario].unique()

# %%
db['unit'].unique()

# %%
for cm in db[climatemodel].unique():
    print(cm+' has the following model:')
    print(db.filter(climatemodel=cm)['model'].unique())

# %%
db.filter(climatemodel='OSCARv3.0', variable='Effective Radiative Forcing|Anthropogenic', scenario=scenarios_fl).head(20)

# %% [markdown]
# ## Unify the units
# Make sure all models have the same, correct units:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import ar6_ch6_rcmipfigs.utils.misc_func as misc_func  # misc_func

DATA_PROTOCOL = os.path.join(
    INPUT_DATA_DIR,
    "data",
    "submission-template",
    "rcmip-data-submission-template.xlsx",
)
protocol_variables = misc_func.get_protocol_vars(DATA_PROTOCOL)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

protocol_scenarios = misc_func.get_protocol_scenarios(DATA_PROTOCOL)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
#from ar6_ch6_rcmipfigs.utils.misc_func import unify_units
from ar6_ch6_rcmipfigs.utils.plot import plot_available_out

#db_converted_units = unify_units(db, protocol_variables)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
db.head()

# %% [markdown]
# and for these the subcategories can be added up to the total:
#
# Sum up subcategories of
#
# erf_aerosols = "Effective Radiative Forcing|Anthropogenic|Aerosols"
#
# erf_HFC = "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC"
#

# %% [markdown]
# ## Aggregate and rename

# %%
from ar6_ch6_rcmipfigs.utils.misc_func import aggregate_variable

# %%
erf_aerosols = "Effective Radiative Forcing|Anthropogenic|Aerosols"
db_aggregated = db.copy()
for cmod in db_aggregated[climatemodel].unique():
    db_aggregated = aggregate_variable(db_aggregated, erf_aerosols, cmod)  # "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC")
# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
erf_HFC = "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC"
# aggregate HFC variables
for cmod in db_aggregated[climatemodel].unique():
    db_aggregated = aggregate_variable(db_aggregated, erf_HFC, cmod)  # "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC")
# )

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# aggregate Aerosols:
db_aggregated.filter(
    variable=erf_HFC  # "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC"
).head()

# %% [markdown]
# ## Rename variable:

# %%
wrong_name = 'Effective Radiative Forcing|Anthropogenic|Albedo Change|Other|Deposition of Black Carbon on Snow'
right_name = 'Effective Radiative Forcing|Anthropogenic|Other|BC on Snow'
print('Wrong name in climatemodels:')
print(db_aggregated.filter(variable=wrong_name)[climatemodel].unique())
print('Right name in climatemodels:')
print(db_aggregated.filter(variable=right_name)[climatemodel].unique())

# %%
_db = db_aggregated.timeseries().reset_index()
_db[variable] = _db[variable].apply(lambda x: right_name if x == wrong_name else x)
_db=ScmDataFrame(_db)
db_aggregated = _db
print('Wrong name in climatemodels:')
print(db_aggregated.filter(variable=wrong_name)[climatemodel].unique())
print('Right name in climatemodels:')
print(db_aggregated.filter(variable=right_name)[climatemodel].unique())

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
db_aggregated['model'].unique()

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# Don't need this information:
db_aggregated['model'] = ''
db_aggregated['model'].unique()

# %% [markdown]
# ## Available input overview

# %% [markdown]
# The plot below shows which models we have data for:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_available_out(db_aggregated, variables_erf_comp, scenarios_fl)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_available_out(db_aggregated, variables_erf_tot, scenarios_fl, figsize=[25, 10])

# %% [markdown]
# ### Pick out the existing scenarios, models, forcings and times

# %%
print('Variables erf components:', variables_erf_comp)
print()
print('Variables erf totals:', variables_erf_tot)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
db_aggregated = db_aggregated.filter(variable=variables_erf_comp + variables_erf_tot)

scenario = 'scenario'
climatemodel = 'climatemodel'
forcer = 'forcer'
time = 'time'
scenarios = list(db_aggregated[scenario].unique())  # scenarios_fl
climatemodels_fl = list(db_aggregated[climatemodel].unique())
forcings = list(db_aggregated[variable].unique())
times = db_aggregated.time_points  # timeseries().transpose().index

# %% [markdown]
# ## Convert data to xarray dataset
#
#

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import cftime

t_coord = db_aggregated.timeseries().transpose().index.values

ds = xr.Dataset()  # coords={time:t_coord, climatemodel:climatemodels_fl,
#      scenario:scenarios})
first = True
for var in variables_erf_comp + variables_erf_tot:
    # get data array for variable:
    _da = db_aggregated.filter(variable=var, climatemodel=climatemodels_fl
                               ).timeseries().transpose().unstack().to_xarray().squeeze()
    # convert to dataset:
    _ds = _da.to_dataset(name=var)
    # remove coordinate for variabel (contained in name):
    del _ds.coords[variable]
    # merge with existing dataset:
    ds = xr.merge([_ds, ds])
ds['year'] = xr.DataArray([t.year for t in ds['time'].values], dims='time')
ds['month'] = xr.DataArray([t.month for t in ds['time'].values], dims='time')
ds['day'] = xr.DataArray([t.day for t in ds['time'].values], dims='time')
# Convert to cftime
dates = [cftime.DatetimeGregorian(y, m, d) for y, m, d in zip(ds['year'], ds['month'], ds['day'])]
ds['time'] = dates
ds = ds.sel(time=slice('1850', '2100'))
ds['time'] = pd.to_datetime([pd.datetime(y, m, d) for y, m, d in zip(ds['year'], ds['month'], ds['day'])])
# Timestep for integral:
ds['delta_t'] = xr.DataArray(np.ones(len(ds['time'])), dims='time', coords={'time': ds['time']})
ds_save = ds.copy()

# %%
ds


# %% [markdown]
# ### Save dataset:

# %%
ds_save.to_netcdf(SAVEPATH_DATASET)

# %%
SAVEPATH_DATASET
