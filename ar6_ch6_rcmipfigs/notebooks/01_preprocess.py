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

import matplotlib.pyplot as plt
import pandas as pd

# %%
from ar6_ch6_rcmipfigs import constants

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# Load data:

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
path_ssps = constants.INPUT_DATA_DIR / 'SSPs'
paths = path_ssps.glob('*')  # '^(minor).)*$')
files = [x for x in paths if x.is_file()]
files

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR

SAVEPATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'
# just minorGHGs_data here
SAVEPATH_DATASET_minor = OUTPUT_DATA_DIR / 'ERF_minorGHGs_data.nc'
SAVEPATH_DATASET

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
ERFs = {}
ERFs_minor = {}
nms = []
for file in files:
    fn = file.name  # filename

    _ls = fn.split('_')  # [1]
    nm = _ls[1]
    print(nm)
    print(file)
    if 'minorGHGs' in fn:
        ERFs_minor[nm] = pd.read_csv(file, index_col=0).copy()
    else:
        ERFs[nm] = pd.read_csv(file, index_col=0).copy()
    nms.append(nm)

# %%
ERFs_minor[nm]

# %%
ERFs[nm].plot()

# %%
for scn in ERFs.keys():
    ERFs[scn]['total_anthropogenic'].plot(label=scn)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', )

# %%
ERFs['ssp534-over'].columns  # [scn]#.columns

# %% [markdown]
# ## Add together aerosol forcing:

# %%
aero_tot = 'aerosol-total'
aero_cld = 'aerosol-cloud_interactions'
aero_rad = 'aerosol-radiation_interactions'
for scn in ERFs.keys():
    # add together:
    ERFs[scn][aero_tot] = ERFs[scn][aero_cld] + ERFs[scn][aero_rad]

# %%
ERFs['ssp534-over'].columns  # [scn]#.columns

# %% [markdown]
# ## SUM OF HCF

# %%
HFCs_name = 'HFCs'
# list of variables
ls = list(ERFs_minor['ssp119'].columns)
# choose only those with HFC in them
vars_HFCs = [v for v in ls if 'HFC' in v]

vars_HFCs

# %%

# %%
for scn in ERFs_minor.keys():
    # sum over HFC variables
    ERFs_minor[scn][HFCs_name] = ERFs_minor[scn][vars_HFCs].sum(axis=1)
    # add row to ERFs as well
    ERFs[scn][HFCs_name] = ERFs_minor[scn][HFCs_name]
ERFs[scn]

# %%
ERFs.keys()

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import xarray as xr

das = []
for nm in nms:
    # convert to xarray
    ds = ERFs[nm].to_xarray()  # .squeeze()
    # concatubate variables as new dimension
    da = ds.to_array('variable')
    # give scenario name
    da = da.rename(nm)

    das.append(da)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# let the new dimension be called scenario:
da_tot = xr.merge(das).to_array('scenario')
# rename the dataset to ERF
da_tot = da_tot.rename('ERF')
# save
da_tot.to_netcdf(SAVEPATH_DATASET)
da_tot.to_dataset()

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import xarray as xr

das = []
for nm in nms:
    ds = ERFs_minor[nm].to_xarray()  # .squeeze()
    da = ds.to_array('variable')
    da = da.rename(nm)
    das.append(da)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
da_tot_minor = xr.merge(das).to_array('scenario')
da_tot_minor = da_tot_minor.rename('ERF')
da_tot_minor.to_netcdf(SAVEPATH_DATASET_minor)
da_tot_minor.to_dataset()

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
da_check = xr.open_dataset(SAVEPATH_DATASET)
da_check

# %%
import matplotlib.pyplot as plt

# %%
for scn in da_check.scenario:
    da_check.sel(variable='total_anthropogenic')['ERF'].sel(scenario=scn).plot(label=scn.values)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', )

# %%
SAVEPATH_DATASET

# %%
