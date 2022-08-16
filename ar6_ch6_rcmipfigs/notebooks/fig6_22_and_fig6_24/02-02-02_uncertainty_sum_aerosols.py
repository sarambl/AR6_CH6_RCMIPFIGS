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
# # Minor revisions of the uncertainties in future warmings from the FaIR model:

# %% [markdown] tags=[]
# ### Estimate uncertainty for aerosol including BC on snow
#
# BC on snow is not included in the estimated temperature change from aerosols, but in the following we wish to
# include both together. This requires an estimate of the uncertainty of the sum. We estimate sigma for the sum
# following the logic below:

# %% [markdown] tags=[]
# ### HFCs
# The uncertainty estimates for temperature change output from FaIR contains more HFCs than those which are considered short lived. In the final figures we exclude some HFCs for this reason, and thus we also scale the uncertainties with the change in the magnitude of the total sum. 
#

# %% [markdown]
# ## Aerosol + BC on snow

# %% [markdown] tags=[]
# ### Estimate of sigma for bc on snow + aerosol total
# We assume that both variables are normally distributed and independent. We estimate the sigma of each of them as
#
# $\sigma_X = (p84_X -p16_X)/2 $
#
# where $p84_X$ and $p16_X$ are the 84th and 16th percentile. 
#
#

# %% [markdown]
# Let $X$ and $Y$ be independent random variables that are normally distributed (and therefore also jointly so), then their sum is also normally distributed. i.e., if
#
# $X\sim N(\mu_{X},\sigma_{X}^{2})$ 
#
# $Y\sim N(\mu_{Y},\sigma_{Y}^{2})$
#
# and
#
# $Z=X+Y$
#
# then
#
# $Z\sim N(\mu _{X}+\mu _{Y},\sigma _{X}^{2}+\sigma _{Y}^{2})$.

# %%
import matplotlib.pyplot as plt

import xarray as xr
import numpy as np

# %%

# %load_ext autoreload
# %autoreload 2


# %%
aero_tot = 'aerosol-total'
aero_cld = 'aerosol-cloud_interactions'
aero_rad = 'aerosol-radiation_interactions'
bc_on_snow = 'bc_on_snow'
aero_tot_wbc = 'aerosol-total-with_bc-snow'

# %% [markdown]
# ### Load data:

# %% pycharm={"name": "#%%\n"}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR

PATH_DT_INPUT = OUTPUT_DATA_DIR / 'fig6_22_and_6_24'/'dT_data_RCMIP_recommendation.nc'
PATH_DT_INPUT_minor = OUTPUT_DATA_DIR / 'fig6_22_and_6_24' / 'dT_data_RCMIP_recommendation_minor.nc'

PATH_UNCERT_DT_INPUT = OUTPUT_DATA_DIR /'fig6_22_and_6_24' /  'dT_uncertainty_data_FaIR_chris.nc'
PATH_UNCERT_DT_OUTPUT = OUTPUT_DATA_DIR / 'fig6_22_and_6_24' / 'dT_uncertainty_data_FaIR_chris_ed02-02-02.nc'

# %%
ds_unc = xr.open_dataset(PATH_UNCERT_DT_INPUT)

# %%
ds_unc

# %%
v_sigma2_bot = 'p16-p50'
v_sigma2_top = 'p84-p50'
np.abs(ds_unc[v_sigma2_bot]).sel(scenario='ssp119', variable=bc_on_snow).plot(label='5%')
np.abs(ds_unc[v_sigma2_top]).sel(scenario='ssp119', variable=bc_on_snow).plot(label='95%')
plt.legend()
plt.show()

v_sigma2_bot = 'p16'
v_sigma2_top = 'p84'
v_med = 'p50'
ds_unc[v_sigma2_bot].sel(scenario='ssp119', variable=bc_on_snow).plot(label='5%')
ds_unc[v_sigma2_top].sel(scenario='ssp119', variable=bc_on_snow).plot(label='95%')
ds_unc[v_med].sel(scenario='ssp119', variable=bc_on_snow).plot(label='median')
plt.legend()
plt.show()

# %% [markdown]
# ### Calculate new distribution of the sum of aerosol + BC on snow, and estimate 5th and 95th percentile:

# %% [markdown]
# Estimate sigma from percentiles

# %%
p05 = 'p05'
p95 = 'p95'
p16 = 'p16'

p84 = 'p84'
median = 'p50'
es_sig = 'estimated_sigma'
es_median = 'estimated_median'
ds_unc[es_sig] = np.abs((ds_unc[p16] - ds_unc[p84]) / 2)

# %% [markdown]
# Compute sigma for sum: 

# %%
ds_bc_snow = ds_unc.sel(variable=bc_on_snow)
ds_aero_tot = ds_unc.sel(variable=aero_tot)
sigma_sum = np.sqrt(ds_bc_snow[es_sig] ** 2 + ds_aero_tot[es_sig] ** 2)
median_sum = ds_bc_snow[median] + ds_aero_tot[median]
_coords = ds_bc_snow.coords
_coords['variable'] = [aero_tot_wbc]
ds_sum = xr.Dataset(coords=_coords)  # {'variable':[aero_tot_wbc]})
# sig_bc_snow.coords)
ds_sum

# %% [markdown]
# Estimate percentiles for new distribution (assumed normal): 

# %%
from scipy.stats import norm

di = norm(loc=median_sum, scale=sigma_sum)

ds_sum[p05] = xr.DataArray(di.ppf(0.05), coords=ds_bc_snow[p05].coords)
ds_sum[p95] = xr.DataArray(di.ppf(0.95), coords=ds_bc_snow[p05].coords)
ds_sum[p16] = xr.DataArray(di.ppf(0.16), coords=ds_bc_snow[p05].coords)
ds_sum[p84] = xr.DataArray(di.ppf(0.84), coords=ds_bc_snow[p05].coords)
ds_sum[median] = xr.DataArray(di.ppf(0.50), coords=ds_bc_snow[p05].coords)

# %%
for v in [p05, p16, p84, p95, median]:
    ds_sum[f'{v}-{median}'] = ds_sum[v] - ds_sum[median]

# %% [markdown]
# ### Plot result: 

# %%
cols = {p05: 'r', p95: 'b', median: 'g'}
for c in [p05, p95, median]:
    ds_bc_snow[c].sel(scenario='ssp119').plot(c=cols[c], label=c + ', bc', linestyle='dashed')
    ds_aero_tot[c].sel(scenario='ssp119').plot(c=cols[c], label=c + ', aero', linestyle='dotted')
    ds_sum[c].sel(scenario='ssp119').plot(c=cols[c], linestyle='solid', label=c + ', sum')

# ds_aero_tot[p95].sel(scenario='ssp119').plot()
# ds_aero_tot[median].sel(scenario='ssp119').plot()
plt.legend()
plt.show()

# %% [markdown]
# ### Write to netcdf

# %%
# noinspection PyDeprecation
ds_unc_upd = xr.concat([ds_unc.drop([es_sig]), ds_sum], dim='variable')
ds_unc_upd.variable
ds_unc_upd.to_netcdf(PATH_UNCERT_DT_OUTPUT)

# %% [markdown] tags=[]
# ## HFCs
# The uncertainty estimates for temperature change output from FaIR contains more HFCs than those which are considered short lived. In the final figures we exclude some HFCs for this reason, and thus we also scale the uncertainties with the change in the magnitude of the total sum. 
#

# %%
# excluded_HFCs = ['HFC-23','HFC-125','HFC-143a','HFC-227ea','HFC-236fa']
excluded_HFCs = ['HFC-23', 'HFC-236fa']  # 'HFC-125','HFC-227ea','HFC-143a',

# %% [markdown]
# HFCs ordered by lifetime: 

# %%
ordered_lifetime_ls = [
    'HFC-152a',
    'HFC-32',
    'HFC-245fa',
    'HFC-365mfc',
    'HFC-134a',
    'HFC-43-10mee',
    'HFC-125',
    'HFC-227ea',
    'HFC-143a',
    'HFC-236fa',
    'HFC-23',
]

hfcs_tau = {
    'HFC-152a': 1.6,
    'HFC-32': 5.4,
    'HFC-245fa': 7.9,
    'HFC-365mfc': 8.9,
    'HFC-134a': 14.0,
    'HFC-43-10mee': 17.0,
    'HFC-125': 30.0,
    'HFC-227ea': 36.0,
    'HFC-143a': 51.0,
    'HFC-236fa': 213.0,
    'HFC-23': 228,
}

# %%
name_deltaT = 'Delta T'

# %% [markdown]
# ### Load data

# %% [markdown]
# Open dataset:

# %%
ds_DT = xr.open_dataset(PATH_DT_INPUT)
ds_DT_minor = xr.open_dataset(PATH_DT_INPUT_minor)

# hfc Delta T
hfc_dT = ds_DT_minor[name_deltaT].sel(percentile='recommendation')

# %%
hfc_dT

# %% [markdown]
# Extract HFCs in file

# %% [markdown]
# ### Compute ratio of sum of short lived HFCs and all HFCs: 

# %%
ls = list(ds_DT_minor['variable'].values)
# chocose only those with HFC in them
vars_HFCs = [v for v in ls if 'HFC-' in v]

vars_HFCs

# %% [markdown]
# List of HFCs without the excluded ones:

# %%
final_HFC_vars = [hfc for hfc in vars_HFCs if hfc not in excluded_HFCs]

final_HFC_vars

# %% [markdown]
# Sum of only short lived HFCs (excluding the longer lived ones):  

# %%
hfs_sum_exclusive = hfc_dT.sel(variable=final_HFC_vars).sum('variable')

# %% [markdown]
# Sum with all HFCs: 

# %%
hfs_sum_all = hfc_dT.sel(variable=vars_HFCs).sum('variable')

# %% [markdown]
# Ratio between sum of short lived and total: 

# %%
rat = hfs_sum_exclusive / hfs_sum_all

# %% [markdown]
# ### Plots:

# %% [markdown]
# #### Plot of ratio between short lived and total sum of HFCs over time:

# %%
for scn in hfs_sum_all['scenario'].values:
    rat.sel(scenario=scn, year=slice(2020, 2100)).plot(label=scn)
    # hfs_sum_all.sel(scenario=scn).plot(label=scn)
    # hfs_sum_exclusive.sel(scenario=scn).plot(label=scn)

plt.legend()
plt.show()

# %% [markdown]
# #### Plot of temperature change over time for each HFC in example scenario

# %%
for hf in ordered_lifetime_ls:
    hfc_dT.sel(variable=hf, scenario='ssp334', year=slice(2000, 2100)).plot(label=hf + ', %s' % hfcs_tau[hf])
plt.legend()
plt.show()

# %% [markdown]
# #### Plot of 95th percentile minus median for origial

# %%
_da = ds_unc_upd['p95-p50'].sel(variable='HFCs')
for scn in _da['scenario'].values:
    _da.sel(scenario=scn).plot()
plt.show()

# %% [markdown]
# ### Scale uncertainty in sum of HFCs by ratio between only short lived HFCs and all:

# %% [markdown]
# Copy input:

# %%
ds_unc_upd2 = ds_unc_upd.copy(deep=True)

# %% [markdown]
# Update values of uncertainty of HFCs by scaling the difference between percentile and median according to the change in total HFCs: 

# %%
v = 'p05-p50'
_tmp = (ds_unc_upd2[v]).loc[{'variable': 'HFCs'}] * rat
ds_unc_upd2[v].loc[{'variable': 'HFCs'}] = _tmp

# %%
v = 'p95-p50'
_tmp = (ds_unc_upd2[v]).loc[{'variable': 'HFCs'}] * rat
ds_unc_upd2[v].loc[{'variable': 'HFCs'}] = _tmp

# %%
ds_unc_upd2[v].sel(scenario='ssp585', variable='HFCs').plot()

ds_unc_upd[v].sel(scenario='ssp585', variable='HFCs').plot()

# %% [markdown]
# ### Write to netcdf:

# %%
ds_unc_upd2.to_netcdf(PATH_UNCERT_DT_OUTPUT)

# %%
print(PATH_UNCERT_DT_OUTPUT)

# %%
