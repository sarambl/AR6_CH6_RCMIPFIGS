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
# # Compute change in temperature from ERF

# %% [markdown]
# This notebook takes the ERF dataset created in notebook 1 and computes the change in temperature from each forcing according to an impulse response function

# %% [markdown]
# ### Imports
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
import xarray as xr
from IPython.display import clear_output
# %%
from openscm_twolayermodel import ImpulseResponseModel  # pip install openscm-twolayermodel
from openscm_units import unit_registry  # pip install openscm-units
from scmdata import ScmRun  # pip install scmdata

# %load_ext autoreload
# %autoreload 2
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR_BADC
# %%
from ar6_ch6_rcmipfigs.utils.badc_csv import read_csv_badc

# %% [markdown]
# ### General about computing $\Delta T$:
# %% [markdown]
# We compute the change in GSAT temperature ($\Delta T$) from the effective radiative forcing (ERF) from MAGICC?????? (#TODO: check model and reference), by integrating with the impulse response function (IRF(t-t'))
#
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
# Where the constants, $q_i$ and $d_i$ are shown below. 
#
# **TODO: find ref**
# %% [markdown]
# ## Input data:
# See [README.md](../../README.md)
# %% [markdown]
# # Code + figures

# %%
fn_IRF_constants = INPUT_DATA_DIR_BADC /'recommended_irf_from_2xCO2_2021_02_25_222758.csv'

irf_consts = read_csv_badc(fn_IRF_constants).set_index('id')

ld1 = 'd1 (yr)'
ld2 = 'd2 (yr)'
lq1 = 'q1 (K / (W / m^2))'
lq2 = 'q2 (K / (W / m^2))'
median = 'median'
perc5 = '5th percentile'
perc95 = '95th percentile'
recommendation = 'recommendation'
irf_consts  # [d1]

# %%
# lets get the irf values from 0 until i
d1 = float(irf_consts[ld1])
d2 = float(irf_consts[ld2])
q1 = float(irf_consts[lq1])
q2 = float(irf_consts[lq2])
eff = float(irf_consts['efficacy (dimensionless)'])
print(f'd1={d1}, d2={d2}, q1={q1}, q2={q2}')


# %%

print(f'd1={d1}, d2={d2}, q1={q1}, q2={q2}')


# %% [markdown]
# ### Path input data

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

PATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'
PATH_DATASET_minor = OUTPUT_DATA_DIR / 'ERF_minorGHGs_data.nc'







# %% [markdown]
# ## Path output data

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
PATH_DT_TAB_OUTPUT = RESULTS_DIR / 'tables' / 'table_sens_dT_cs_recommandetion.csv'
#PATH_DT_OUTPUT = OUTPUT_DATA_DIR / 'dT_data_RCMIP_recommendation.nc'
PATH_DT_OUTPUT = OUTPUT_DATA_DIR / 'dT_data_RCMIP_recommendation_minor.nc'

print(PATH_DT_OUTPUT)


# %% [markdown]
# ## various definitions

# %%
# name of output variable
name_deltaT = 'Delta T'

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'
percentile = 'percentile'

# %% [markdown]
# ## Set values:

# %% [markdown]
# We only compute for the recommendation (not e.g. percentiles):

# %%
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
scenarios_fl = ['ssp534-over', 'ssp119', 'ssp334','ssp460', 'ssp585', 'ssp370',
                'ssp370-lowNTCF-aerchemmip', 'ssp126', 'ssp245', 'ssp434',
                'ssp370-lowNTCF-gidden'
                ]


# %% [markdown]
# ## IRF function: 

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
# ### Open ERF dataset:

# %%
ds = xr.open_dataset(PATH_DATASET).sel(year=slice(1700, 2200))  # we need only years until 1700
da_ERF = ds['ERF']

# %%
ds_minor = xr.open_dataset(PATH_DATASET_minor).sel(year=slice(1700, 2200))  # we need only years until 1700
da_ERF_minor = ds['ERF']

# %%
da_ERF_minor.sel(variable='HFCs').isel(scenario=1)

# %% [markdown]
# #### Simple pre-processing

# %%
ds['time'] = pd.to_datetime(ds['year'].to_pandas().index.map(str), format='%Y')
ds_minor['time'] = pd.to_datetime(ds_minor['year'].to_pandas().index.map(str), format='%Y')

# delta_t is 1 (year)
ds['delta_t'] = xr.DataArray(np.ones(len(ds['year'])), dims='year', coords={'year': ds['year']})
ds_minor['delta_t'] = xr.DataArray(np.ones(len(ds_minor['year'])), dims='year', coords={'year': ds['year']})

# %% [markdown]
# ## Integrate and compute $\Delta T$:
# The code below integrates the read in ERFs with the pre defined impulse response function (IRF).

# %% [markdown]
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %%


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
    #fname = OUTPUT_DATA_DIR/ fn#'DT_%s-%s.nc' % (from_t, to_t)
    # save dataset.
    #ds_DT.to_netcdf(fname)
    return ds_DT


# %%

# %%
int(ds.isel(year=0)['year'].values)


# %%
def calc_dGSAT(var, ds, ds_out, scenario='scenario'):
    s_y = int(ds.isel(year=0)['year'].values)
    _erf_tmp = ds['ERF'].sel(variable=var).to_pandas()
    unit = "W/m^2"

    driver = ScmRun(

        data=_erf_tmp,
        index=s_y + np.arange(len(_erf_tmp)),
        columns={
            "unit": unit,
            "model": "custom",
            "scenario": scenario,
            "region": "World",
            "variable": "Effective Radiative Forcing",
        },
    )

    impulse_res = ImpulseResponseModel(
        d1=d1 * unit_registry("yr"),
        d2=d2 * unit_registry("yr"),
        q1=q1* unit_registry("delta_degC / (W / m^2)"),
        q2=q2* unit_registry("delta_degC / (W / m^2)"),
        efficacy=eff* unit_registry("dimensionless"),
    )
    dt_tmp = impulse_res.run_scenarios(driver)


    df_tmp= dt_tmp.filter(variable='Surface Temperature').timeseries()#.lineplot()#['Surface']
    #_ds_dT[var] =df_tmp.transpose()

    #ds_out[var]  =
    df_tmp = df_tmp.reset_index().iloc[:,12:].transpose().rename({0:var}, axis=1)#.to_xarray()
    year_index = pd.to_datetime(df_tmp.index).year
    df_tmp['year'] = year_index
    df_tmp = df_tmp.set_index('year')

    ds_out[var] = df_tmp.to_xarray()[var]

    return ds_out


# %%
def calc_GSAT_all_vars(_ds, ds_out, variables=None, scenario='scenario'):


    if variables is None:
        variables =_ds['variable'].values


    _ds_dT = xr.Dataset()
    for var in variables:
        print(var)
        _ds_dT = calc_dGSAT(var, _ds, _ds_dT, scenario=scenario)
        print()
    ds_DT = _ds_dT.to_array(name=name_deltaT)

    #ds_out[name_deltaT] = ds_DT


    return ds_DT


# %%
def calc_GSAT_all_scenarios(ds, ds_out, scenarios_l = None):


    scenarios_l = None
    if scenarios_l is None:
        scenarios_l = ds[scenario].values
    _ds_dT = xr.Dataset()
    _ds_out = xr.Dataset()
    for scn in scenarios_l:
        print(scn)
        ds_scn = calc_GSAT_all_vars(ds.sel(scenario = scn), _ds_dT, scenario=scn)
        ds_scn = ds_scn.rename(scn)
        _ds_out[scn] = ds_scn
    ds_DT = _ds_out.to_array(dim=scenario, name=name_deltaT)
    ds_out[name_deltaT] = ds_DT


    return ds_out

# %%
ds_minor

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
dic_minor_ds_old = {}
for key in IRFpercentiles:
    dic_minor_ds_old[key] = integrate_to_dT(ds_minor, first_y, last_y, irf_consts.loc[key], int_var='ERF')

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} tags=[]
dic_minor_ds = {}
ds_out = ds_minor.copy(deep=True)
for key in IRFpercentiles:
    dic_minor_ds[key] = calc_GSAT_all_scenarios(ds_minor,ds_out)

# %% [markdown]
# ## check:

# %%
_ds_minor = dic_minor_ds['recommendation']
_ds_minor['variable'].values

# %%
ls = list(ds_minor['variable'].values)
# chocose only those with HFC in them
vars_HFCs = [v for v in ls if 'HFC' in v]

vars_HFCs

# %%
len(vars_HFCs)

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

# %%
hfcs_tau = {
'HFC-152a':1.6,
'HFC-32':5.4,
'HFC-245fa':7.9,
'HFC-365mfc':8.9,
'HFC-134a':14.0,
'HFC-43-10mee':17.0,
'HFC-125':30.0,
'HFC-227ea':36.0,
'HFC-143a':51.0 ,
'HFC-236fa':213.0,
'HFC-23':228,
}

# %%
ordered_lifetime_ls[:0]

# %%
import numpy as np

# %%

# %%
for i in np.arange(len(ordered_lifetime_ls)):
    _tmp = _ds_minor[name_deltaT].sel(variable=ordered_lifetime_ls[0:(i+1)])
    _tmp.sel(scenario='ssp585').sum('variable').plot(label=f'up to:{ordered_lifetime_ls[i]}, $\\tau$={hfcs_tau[ordered_lifetime_ls[i]]}')
plt.xlim([2000,2100])
plt.legend()
plt.show()

# %%
ordered_lifetime_ls

# %%
rn_dic = {}
for v in ordered_lifetime_ls:
    tau = hfcs_tau[v]
    
    rn_dic[v]=f'{v}, $\\tau$={tau}'

# %%
_tmp = _ds_minor[name_deltaT].sel(scenario='ssp585',year=2100, variable=ordered_lifetime_ls)#\.plot()
_tmp.to_pandas().rename(rn_dic).plot.barh()
plt.title('$\Delta$ GSAT, SSP5-8.5, year 2100')
plt.xlabel('$^\circ$ C')
plt.show()

# %%
for per in IRFpercentiles:
    dic_minor_ds[per].isel(scenario=0, variable=0)[name_deltaT].plot()
plt.show()

# %% [markdown]
# ### Make datset with percentile as dimension:
# Does really only make sense with percentiles...

# %% [markdown]
# ds_tmp = xr.Dataset(coords=dic_ds[recommendation].coords)
# ds_tmp
# for key in IRFpercentiles:
#     ds_tmp[key] = dic_ds[key]['Delta T']  # .dims,dic_ds[key],)
# ds['Delta T'] = ds_tmp.to_array('percentile')

# %%
ds_tmp = xr.Dataset(coords=dic_minor_ds[recommendation].coords)
ds_tmp
for key in IRFpercentiles:
    ds_tmp[key] = dic_minor_ds[key]['Delta T']  # .dims,dic_ds[key],)
ds_minor['Delta T'] = ds_tmp.to_array('percentile')

# %% [markdown]
# # Save dataset:

# %%
ds_minor['Delta T'].year

# %%
ds_minor.sel(year=slice(first_y, last_y)).to_netcdf(PATH_DT_OUTPUT)
print(f'Saved to {PATH_DT_OUTPUT}')
# ds_DT.to_array('percentile')
# dic_ds[key]['Delta T']

# %%
