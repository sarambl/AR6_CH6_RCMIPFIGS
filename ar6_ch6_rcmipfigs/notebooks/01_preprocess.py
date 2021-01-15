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
import glob
from ar6_ch6_rcmipfigs import constants
import pandas as pd
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# Load data:

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
path_ssps = constants.INPUT_DATA_DIR /'SSPs'
paths = path_ssps.glob('*')
files = [x for x in paths if x.is_file()]
files




# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, INPUT_DATA_DIR

SAVEPATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'
SAVEPATH_DATASET

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
ERFs = {}
nms = []
for file in files:
    fn = file.name # filename
    nm = fn.split('_')[1]
    print(nm)
    print(file)
    ERFs[nm] = pd.read_csv(file).copy()
    nms.append(nm)


# %%
ERFs[nm].set_index('year').plot()

# %%
for scn in ERFs.keys(): 
    ERFs[scn].set_index('year')['total_anthropogenic'].plot(label=scn)
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left',)

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
import xarray as xr
das = []
for nm in nms:
    ds = ERFs[nm].set_index('year').to_xarray()#.squeeze()
    da = ds.to_array('variable')
    da = da.rename(nm)
    das.append(da)

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
da_tot = xr.merge(das).to_array('scenario')
da_tot = da_tot.rename('ERF')

da_tot.to_netcdf(SAVEPATH_DATASET)
da_tot.to_dataset()

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
da_check = xr.open_dataset(SAVEPATH_DATASET)
da_check


# %%
import matplotlib.pyplot as plt

# %%
for scn in da_check.scenario: 
    da_check.sel(variable='total_anthropogenic')['ERF'].sel(scenario=scn).plot(label=scn.values)
plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left',)

# %%
SAVEPATH_DATASET
