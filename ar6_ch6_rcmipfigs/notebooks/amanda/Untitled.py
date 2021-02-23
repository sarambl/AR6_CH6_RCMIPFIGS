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
PATH_DT_UNCERTAINTY = OUTPUT_DATA_DIR / 'dT_uncertainty_data_FaIR_chris_ed02-3.nc'


# %% [markdown]
# ## Set values:

# %%
base_period = [1995,2014]
short_term = [2021,2040]
long_term = [2041,2060]
vlong_term = [2081,2100]
periods = [short_term, long_term,vlong_term]
periods_dic  = {
    'near-term':short_term,
    'mid-term':long_term,
    'long-term':vlong_term
}

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = '2020'

# %%
FIGURE_DIR = RESULTS_DIR / 'figures_amanda/'

TABLE_DIR = RESULTS_DIR / 'tables_amanda/'

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
sum_SLCF = 'Sum SLCF (Aerosols, Methane, Ozone, HFCs)'
name_deltaT = 'Delta T'

# %% [markdown]
# ### Define variables to look at:

# %%
# variables to plot:
variables_erf_comp = [
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
                'ssp334',
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


# %%

# %%
ds_DT = xr.open_dataset(PATH_DT)
ds_uncertainty = xr.open_dataset(PATH_DT_UNCERTAINTY)

# %%
ds_DT.scenario  # .climatemodel

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_var_nicename

# %%
_str = ''
_vl = [get_var_nicename(var).split('(')[0].strip() for var in variables_erf_comp]
for var in _vl: 
    _str += f'{var}, '

# ax.set_title('Temperature change, sum SLCF  (%s)' % _str[:-2])


vn_sum = 'Sum SLCF (%s)' % _str[:-2]
print(vn_sum)
#_st = vn_sum.replace('(','').replace(')','').replace(' ','_').replace(',','')+'.csv'

_da_sum  = ds_DT[name_deltaT].sel(variable=variables_erf_comp).sum(variable)
#_da = ds_DT[name_deltaT].sel(variable=variables_erf_comp).sum(variable).sel(year=slice(int(s_y2), int(e_y2))) - ds_DT_sy
_da_sum#.assin_coord()
#_ds_check = ds_DT.copy()
ds_DT
#xr.concat([_ds_check[name_deltaT],_da_sum], dim=variable)

dd1=_da_sum.expand_dims(
    {'variable':
    [vn_sum]})
#dd1=dd1.to_dataset()

ds_DT = xr.merge([ds_DT,dd1])

# %%
base_mean = ds_DT.sel(year=slice(*base_period)).mean('year')#base_period[0], base_period[1]))
bp_str = f'({base_period[0]}-{base_period[1]})'

per_means = []
per_means_anomaly = []
per_names = []
for per_n in periods_dic.keys():
    per = periods_dic[per_n]
    print(per)
    per_mean = ds_DT.sel(year=slice(*per)).mean('year')
    per_means.append(per_mean)
    per_anom = per_mean- base_mean
    anom_df = per_anom['Delta T'].squeeze().to_dataframe().drop('percentile', axis=1)
    #per_names.append()
    lab = f'{per_n} ({per[0]}-{per[1]})-({base_period[0]}-{base_period[1]})'
    per_means_anomaly.append(anom_df.rename({name_deltaT:lab}, axis=1))
    
short_term_mean = ds_DT.sel(year=slice(*short_term)).mean('year')
long_term_mean = ds_DT.sel(year=slice(*long_term)).mean('year')
base_mean#['year']

# %%
import pandas as pd

# %%
output_df = pd.concat(per_means_anomaly, axis=1)


# %%
fn = TABLE_DIR / 'period_anomalies.csv'
output_df.to_csv(fn)
output_df

# %% [markdown]
# per_anom['Delta T'].to_dataframe()#.drop('percentile', axis=1)
#
#
#
# st_anomaly = short_term_mean-base_mean
# lt_anomaly = long_term_mean- base_mean
# st_anomaly_df = st_anomaly.squeeze('percentile')['Delta T'].to_dataframe().drop('percentile', axis=1)
# lt_anomaly_df = lt_anomaly.squeeze('percentile')['Delta T'].to_dataframe().drop('percentile', axis=1)
#
# bp_str = f'({base_period[0]}-{base_period[1]})'
# st_str = f'({short_term[0]}-{short_term[1]})'
# lt_str = f'({long_term[0]}-{long_term[1]})'
#
# import pandas as pd
#
# st_anomaly_df = st_anomaly_df.rename({name_deltaT: f'short term anomaly ({st_str}-{bp_str})'}, axis=1)
# lt_anomaly_df = lt_anomaly_df.rename({name_deltaT: f'long term anomaly ({lt_str}-{bp_str})'}, axis=1)
# output_df = pd.concat([st_anomaly_df, lt_anomaly_df], axis=1)
#
# ds_DT.sel(year=slice(*short_term))['year']
