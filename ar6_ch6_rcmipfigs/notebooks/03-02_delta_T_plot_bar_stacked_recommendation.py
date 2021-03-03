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
# # Plot temperature response over time

# %% [markdown]
# This notebook plots temperature respons to SLCFs AND the total scenario forcing in a fixed nr of years

# %% [markdown]
# ## Imports:
#
# import matplotlib.pyplot as plt
# import pandas as pd
# %%
import matplotlib.pyplot as plt

# %%
import xarray as xr
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# %load_ext autoreload
# %autoreload 2

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_cmap_dic
from ar6_ch6_rcmipfigs.utils.plot import get_var_nicename

# %%
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

# PATH_DATASET = OUTPUT_DATA_DIR + '/forcing_data_rcmip_models.nc'
# PATH_DT = OUTPUT_DATA_DIR / '/dT_data_rcmip_models.nc'
# PATH_DT = OUTPUT_DATA_DIR / 'dT_data_RCMIP.nc'
PATH_DT = OUTPUT_DATA_DIR / 'dT_data_RCMIP_recommendation.nc'

# %% [markdown]
# #### Uncertainty data from Chris

# %%
# PATH_DT_UNCERTAINTY = OUTPUT_DATA_DIR / 'dT_uncertainty_data_FaIR_chris.nc'
PATH_DT_UNCERTAINTY = OUTPUT_DATA_DIR / 'dT_uncertainty_data_FaIR_chris_ed02-3.nc'


# %% [markdown]
# ## Set values:

# %%
first_y = '1750'
last_y = '2100'
# Years to plot:
years = ['2040', '2100']

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = '2019'
ref_year_uncertainty = '2020'

# %%
FIGURE_DIR = RESULTS_DIR / 'figures_recommendation/'

TABS_DIR = RESULTS_DIR / 'tables_recommendation/'

# %%

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'

# %%
recommendation = 'recommendation'
name_deltaT = 'Delta T'
sum_v = 'Sum SLCF (Aerosols, Methane, Ozone, HFCs)'

scenario_tot = 'Scenario total'

# %% [markdown]
# ### Define variables to look at:

# %%
variables_erf_comp = [
    'aerosol-total-with_bc-snow',
    'ch4',
    'o3',
    'HFCs',
]

# total ERFs for anthropogenic and total:
variables_erf_tot = ['total_anthropogenic',
                     'total']
variables_all = variables_erf_comp + variables_erf_tot
# Scenarios to plot:

# %%

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
                'ssp370-lowNTCF-gidden',
                'ssp585']
scenarios_fl_oneNTCF = ['ssp119',
                        'ssp126',
                        'ssp245',
                        'ssp370',
                        'ssp370-lowNTCF-aerchemmip',
                        # 'ssp370-lowNTCF-gidden',
                        'ssp585']

# %%
median = 'median'
perc5 = '5th percentile'
perc95 = '95th percentile'

# %%
table_csv_name = TABS_DIR / '3-2_table_all_scen.csv'
print(table_csv_name)

# %%

variables_tot = ['Total']
variables_sum = ['Sum SLCFs']


def setup_table_prop(scenario_n='', yrs=None, _vlist=None, scens=None):
    if _vlist is None:
        _vlist = variables_erf_comp
    if yrs is None:
        yrs = ['2040', '2100']
    if scens is None:
        scens = scenarios_fl
    its = [yrs, _vlist]
    _i = pd.MultiIndex.from_product(its, names=['', ''])
    table = pd.DataFrame(columns=scens, index=_i)  # .transpose()
    table.index.name = scenario_n
    return table


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
print(PATH_DT)

# %% [markdown]
# ## Add sum as variable:

# %%
_str = ''
_vl = [get_var_nicename(var).split('(')[0].strip() for var in variables_erf_comp]
for var in _vl:
    _str += f'{var}, '

# ax.set_title('Temperature change, sum SLCF  (%s)' % _str[:-2])


vn_sum = 'Sum SLCF (%s)' % _str[:-2]
print(vn_sum)

# _st = vn_sum.replace('(','').replace(')','').replace(' ','_').replace(',','')+'.csv'


_da_sum = ds_DT[name_deltaT].sel(variable=variables_erf_comp).sum(variable)
# _da = ds_DT[name_deltaT].sel(variable=variables_erf_comp).sum(variable).sel(year=slice(int(s_y2), int(e_y2))) - ds_DT_sy
_da_sum  # .assin_coord()
# _ds_check = ds_DT.copy()
ds_DT
# xr.concat([_ds_check[name_deltaT],_da_sum], dim=variable)

dd1 = _da_sum.expand_dims(
    {'variable':
         ['Sum SLCF (Aerosols, Methane, Ozone, HFCs)']})
# dd1=dd1.to_dataset()

ds_DT = xr.merge([ds_DT, dd1])

# %% [markdown]
# ## Compute sum of all SLCF forcers


# %%
from ar6_ch6_rcmipfigs.utils.plot import get_scenario_c_dic, get_scenario_ls_dic

# %%


cdic = get_scenario_c_dic()
lsdic = get_scenario_ls_dic()


def sum_name(_var):
    """
    Returns the name off the sum o
    """
    return f'{_var} sum '


# %% [markdown]
# ### compute sum: 

# %% [markdown]
# sum_name = 'Sum SLCFs'
# ds_sub = ds_DT.sel(variable=variables_erf_comp)
# ds_sum = ds_sub.sum(variable)
# ds_sum = ds_sum.assign_coords(coords={variable: sum_name})
# # add sum to variable coordinatem
#
# ds_sum = xr.concat([ds_sum, ds_DT.sel(variable=variables_erf_comp)], dim=variable)

# %%
ds_sum = ds_DT  # .assign_coords(coords={variable:sum_name})

rn_dic = {}
for v in variables_all:
    rn_dic[v] = get_var_nicename(v)

from ar6_ch6_rcmipfigs.utils.plot import scn_dic

rn_dic_scen = scn_dic


# %%
def fix_names(df):
    df = df.rename(rn_dic)
    df = df.rename(rn_dic_scen, axis=1)
    return df
# %%
scntab_dic = {}


# tab_tot = setup_table2()
# tab_tot_sd = setup_table2()
def table_of_sts(_ds, _scn_fl, variables, tab_vars, _yrs, ref_yr, sts=recommendation):
    """
    Creates pandas dataframe of statistics (mean, median, standard deviation) for change
    in temperature Delta T since year (ref year) for each scenario in scenarios,

    :param _ds:
    :param _scn_fl:
    :param variables:
    :param tab_vars:
    :param _yrs:
    :param ref_yr:
    :param sts:
    :return:
    """
    tabel = setup_table_prop(yrs=_yrs, _vlist=tab_vars)
    for scn in _scn_fl:
        for _var, tabvar in zip(variables, tab_vars):
            # dtvar =  name_deltaT # if ERF name, changes it here.
            tabscn = scn  # Table scenario name the same.
            for year in _yrs:
                _da = _ds[name_deltaT].sel(scenario=scn, variable=_var)
                _da_refy = _da.sel(year=slice(ref_yr, ref_yr)).squeeze()  # ref year value
                _da_y = _da.sel(year=slice(year, year))  # year value
                _tab_da = _da_y - _da_refy
                tabel.loc[(year, tabvar), tabscn] = float(_tab_da.sel(percentile=sts).squeeze().values)  # [0]
    return fix_names(tabel)
    # return tabel


def table_of_stats_varsums(_ds, dsvar, tabvar, yrs, ref_yr, sts=recommendation):
    """
    Sums up over dimension 'variable' and creates pandas dataframe of statistics (mean, median, standard deviation) for change
    in temperature Delta T since year (ref year) for each scenario in scenarios. 

    :param tabvar:
    :param dsvar:
    :param _ds:
    :param yrs:
    :param ref_yr:
    :param sts:
    :return:
    """
    tabel = setup_table_prop(yrs=yrs, _vlist=[tabvar])
    da = _ds[name_deltaT].sel(variable=dsvar)

    for scn in scenarios_fl:
        tabscn = scn  # Table scenario name the same.
        for year in yrs:
            _da = da.sel(scenario=scn, percentile=sts)  # , variable = dsvar)
            _da_refy = _da.sel(year=slice(ref_yr, ref_yr)).squeeze()  # ref year value
            _da_y = _da.sel(year=slice(year, year)).squeeze()  # year value
            _tab_da = (_da_y - _da_refy).squeeze()

            # Do statistics over RCMIP models
            tabel.loc[(year, tabvar), tabscn] = float(_tab_da.squeeze().values)  # [0]

    return tabel


# %% [markdown]
# ### Computes statistics:

# %%
ds_uncertainty = ds_uncertainty.to_array('percentile').rename(name_deltaT).to_dataset()
ds_uncertainty

# %%
from ar6_ch6_rcmipfigs.utils.plot import nice_name_var as nice_name_var_dic

# %% [markdown]
# ## Get values:
#

# %%
import pandas as pd

# %%
# Mean:
o3 = 'Ozone (O$_3$)'
ch4 = 'Methane (CH$_4$)'
HFCs = 'HFCs'
_sum = 'CH$_4$+O$_3$+HFCs'
sum_v

# Statistics on Delta T anthropogenic
# Mean
scenario_tot = 'Scenario total'

# %%
tabel_dT_anthrop = table_of_sts(ds_DT, scenarios_fl, ['total_anthropogenic'], [scenario_tot], years, ref_year)
# 5th
tabel_dT_anthrop_5th = -table_of_sts(ds_uncertainty,
                                     scenarios_fl,
                                     ['total_anthropogenic'],
                                     [scenario_tot],
                                     years,
                                     ref_year_uncertainty,
                                     sts='p05-p50'
                                     )
# 95th

tabel_dT_anthrop_95th = table_of_sts(ds_uncertainty, scenarios_fl, ['total_anthropogenic'], [scenario_tot], years,
                                     ref_year_uncertainty,
                                     sts='p95-p50')
tabel_dT_anthrop_95th.loc['2040']
tabel_dT_anthrop_5th.loc['2040']

# %%
# Mean:
tabel_dT_slcfs = table_of_sts(ds_DT, scenarios_fl, variables_erf_comp, variables_erf_comp, years,
                              ref_year)

# Compute sum of SLCFs
_ds = ds_sum.copy()

v_sum = 'Sum SLCF (Aerosols, Methane, Ozone, HFCs)'
ds_uncertainty.sel(year=2040, scenario='ssp119', percentile='p05-p50',
                   variable=v_sum)


# %%

tabel_dT_sum_slcf_5 = - table_of_sts(ds_uncertainty,
                                     scenarios_fl,
                                     [sum_v],
                                     ['Sum SLCFs'],
                                     years,
                                     ref_year_uncertainty,
                                     sts='p05-p50')
tabel_dT_sum_slcf_95 = table_of_sts(ds_uncertainty,
                                    scenarios_fl,
                                    [sum_v],
                                    ['Sum SLCFs'],
                                    years,
                                    ref_year_uncertainty,
                                    sts='p95-p50')

# %%

tabel_dT_slcfs = fix_names(tabel_dT_slcfs)  
tabel_dT_slcfs  


# %% [markdown]
# ## colors

# %%
cdic = get_cmap_dic(variables_erf_comp)  # , palette='colorblind'):
ls = [cdic[key] for key in variables_erf_comp]


# %% [markdown]
# ## Error bars only from model uncertainty
# The following uncertainties assume the ECS has a standard deviation of

# %%
from ar6_ch6_rcmipfigs.utils.plot import scn_dic

# %%

def plt_stacked(fig, axs, tabel_dT_anthrop, tabel_dT_slcfs, tabel_dT_anthrop_5th, tabel_dT_anthrop_95th,
                tabel_dT_sum_slcf_5, tabel_dT_sum_slcf_95):
    tits = ['Change in GSAT in 2040 relative to 2019', 'Change in GSAT in 2100 relative to 2019']
    for yr, ax, tit in zip(years, axs, tits):
        # Pick out year and do various renames:
        # Total antropogenic
        tot_yr = tabel_dT_anthrop.loc[yr].rename(scn_dic, axis=1)
        # Sum SLCFs
        # uncertainty:
        bot = tabel_dT_anthrop_5th.loc[yr].rename(scn_dic, axis=1)
        top = tabel_dT_anthrop_95th.loc[yr].rename(scn_dic, axis=1)
        err = pd.merge(bot, top, how='outer').values
        # Plot bars for anthropopogenic total:
        ax.barh(tot_yr.transpose().index, tot_yr.transpose()[scenario_tot].values,
                color='k',
                xerr=err,
                error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2),
                label='Scenario total', alpha=.2,
                )
        # Plot bars for SLCFs total:

        # Plot stacked plot of components:
        _tab = tabel_dT_slcfs.loc[yr].transpose().rename(
            {'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
        _tab = _tab.rename(scn_dic, axis=0)
        a = _tab.plot(kind='barh', stacked=True, ax=ax, color=ls, legend=(yr != '2040'))  # , grid=True)#stac)
        _t = _tab.sum(axis=1)  # , c=100)#.plot(kind='barh', )
        # ax.scatter(_t, list(_t.reset_index().index), zorder=10, c='w', marker='d')
        # uncertainty:
        bot = tabel_dT_sum_slcf_5.loc[yr].rename(scn_dic, axis=1)
        top = tabel_dT_sum_slcf_95.loc[yr].rename(scn_dic, axis=1)
        err = pd.merge(bot, top, how='outer').values

        ax.errorbar(_t, list(_t.reset_index().index), xerr=err, label='__nolabel__', color='w', fmt='d',
                    linestyle="None")  # ,

        if not yr == '2040':
            ax.legend(frameon=False, ncol=1)  # [l],labels=['Sce!!nario total'], loc = 4)#'lower right')
        # Zero line:
        ax.axvline(0, linestyle='--', color='k', alpha=0.4)
        ax.set_title(tit)
        ax.set_xlabel('Change in GSAT ($^\circ$C)')
        ax.xaxis.set_minor_locator(MultipleLocator(.1))

def fix_axs(axs):
    ax = axs[0]

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(right=False, left=False)  # , color='w')

    ax = axs[1]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelleft=False, right=False, left=False, color='w')
    ax.tick_params(labelright=False, labelleft=False, right=False, left=False, color='w')
    ax.yaxis.set_visible(False)
    plt.tight_layout()

# %%
for tab in [tabel_dT_anthrop, tabel_dT_slcfs]:
    display(tab)

# %%
from matplotlib.ticker import MultipleLocator

# %%
fig, axs = plt.subplots(1, len(years), figsize=[10, 4.4], sharex=False, sharey=True)
plt_stacked(fig, axs, tabel_dT_anthrop, tabel_dT_slcfs, tabel_dT_anthrop_5th, tabel_dT_anthrop_95th,
            tabel_dT_sum_slcf_5, tabel_dT_sum_slcf_95)

fix_axs(axs)

fn = FIGURE_DIR / 'stack_bar_influence_years.png'
plt.tight_layout()
# ax = plt.gca()

plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)
plt.show()

# %%

scen_no_lowNTCF = [scn for scn in scenarios_fl if 'lowNTCF' not in scn]
print(scen_no_lowNTCF)
# %%
subset_scenarios = list(pd.Series(scenarios_fl_oneNTCF).replace(rn_dic_scen))
# %%
tabel_dT_anthrop2 = tabel_dT_anthrop[subset_scenarios]
tabel_dT_slcfs2 = tabel_dT_slcfs[subset_scenarios]
tabel_dT_anthrop2_5th = tabel_dT_anthrop_5th[subset_scenarios]
tabel_dT_anthrop2_95th = tabel_dT_anthrop_95th[subset_scenarios]
tabel_dT_sum_slcf2_5 = tabel_dT_sum_slcf_5[subset_scenarios]
tabel_dT_sum_slcf2_95 = tabel_dT_sum_slcf_95[subset_scenarios]

# %%
tabel_dT_slcfs2

# %% [markdown]
# - Include BC in aerosol total.
# - aaarggh, sorry, a last request, maybe one version with also SSP370lowNTC with and without CH4 decrease (as it was in SOD) and one without and we will see later which one we choose depending on the discussion which will remain in the TS.
# - net values. Send.
# - include total in black.

# %%

fig, axs = plt.subplots(1, len(years), figsize=[10, 4.4], sharex=False, sharey=True)
plt_stacked(fig, axs, tabel_dT_anthrop2, tabel_dT_slcfs2, tabel_dT_anthrop2_5th,
            tabel_dT_anthrop2_95th, tabel_dT_sum_slcf2_5, tabel_dT_sum_slcf2_95)

fix_axs(axs)
fn = FIGURE_DIR / 'stack_bar_influence_years_one_lowNTCF.png'
plt.tight_layout()
ax = plt.gca()

plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

plt.show()
# %%

# %%
# tabel_dT_anthrop2 = tabel_dT_anthrop[scen_no_lowNTCF]
# tabel_dT_slcfs2 = tabel_dT_slcfs[scen_no_lowNTCF]

subset_scenarios = list(pd.Series(scen_no_lowNTCF).replace(rn_dic_scen))
# %%
tabel_dT_anthrop2 = tabel_dT_anthrop[subset_scenarios]
tabel_dT_slcfs2 = tabel_dT_slcfs[subset_scenarios]
tabel_dT_anthrop2_5th = tabel_dT_anthrop_5th[subset_scenarios]
tabel_dT_anthrop2_95th = tabel_dT_anthrop_95th[subset_scenarios]
tabel_dT_sum_slcf2_5 = tabel_dT_sum_slcf_5[subset_scenarios]
tabel_dT_sum_slcf2_95 = tabel_dT_sum_slcf_95[subset_scenarios]

# %%
tabel_dT_slcfs2

# %%

fig, axs = plt.subplots(1, len(years), figsize=[10, 3.8], sharex=False, sharey=True)
plt_stacked(fig, axs, tabel_dT_anthrop2, tabel_dT_slcfs2, tabel_dT_anthrop2_5th,
            tabel_dT_anthrop2_95th, tabel_dT_sum_slcf2_5, tabel_dT_sum_slcf2_95)

fix_axs(axs)

fn = FIGURE_DIR / 'stack_bar_influence_years_no_lowNTCF.png'
plt.tight_layout()
ax = plt.gca()

plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
# tabel_dT_anthrop2 = tabel_dT_anthrop[scen_no_lowNTCF]
# tabel_dT_slcfs2 = tabel_dT_slcfs[scen_no_lowNTCF]

scenario_370 = [sc for sc in scenarios_fl if 'ssp370' in sc]
subset_scenarios = list(pd.Series(scenario_370).replace(rn_dic_scen))
# %%
tabel_dT_anthrop2 = tabel_dT_anthrop[subset_scenarios]
tabel_dT_slcfs2 = tabel_dT_slcfs[subset_scenarios]
tabel_dT_anthrop2_5th = tabel_dT_anthrop_5th[subset_scenarios]
tabel_dT_anthrop2_95th = tabel_dT_anthrop_95th[subset_scenarios]
tabel_dT_sum_slcf2_5 = tabel_dT_sum_slcf_5[subset_scenarios]
tabel_dT_sum_slcf2_95 = tabel_dT_sum_slcf_95[subset_scenarios]

# %%

fig, axs = plt.subplots(1, len(years), figsize=[10, 2.5], sharex=False, sharey=True)

plt_stacked(fig, axs, tabel_dT_anthrop2, tabel_dT_slcfs2, tabel_dT_anthrop2_5th,
            tabel_dT_anthrop2_95th, tabel_dT_sum_slcf2_5, tabel_dT_sum_slcf2_95)

fix_axs(axs)

fn = FIGURE_DIR / 'stack_bar_influence_years_horiz_errTot_370only.png'

plt.tight_layout()
ax = plt.gca()

plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
variables_erf_comp

# %%
variables_erf_comp_nbc = ['ch4', 'aerosol-total-with_bc-snow', 'o3', 'HFCs']

# %%
tabel_dT_anthrop

# %%
subset_scenarios

# %%
subset_scenarios_nn =   subset_scenarios

# %%
subset_scen_fl = list(pd.Series(scenarios_fl_oneNTCF).replace(rn_dic_scen))
tabel_dT_slcfs_noBC = table_of_sts(ds_DT, scenarios_fl_oneNTCF, variables_erf_comp_nbc, variables_erf_comp_nbc,
                                   years,
                                   ref_year)

# %%
subset_scenarios = subset_scen_fl
tabel_dT_anthrop2 = tabel_dT_anthrop[subset_scenarios]
tabel_dT_slcfs2 = tabel_dT_slcfs_noBC[subset_scenarios]

# %%
subset_scen_fl = list(pd.Series(scenarios_fl).replace(rn_dic_scen))
tabel_dT_slcfs2 = table_of_sts(ds_DT, scenarios_fl, variables_erf_comp_nbc, variables_erf_comp_nbc, years,
                               ref_year)
tabel_dT_slcfs2 = tabel_dT_slcfs2.rename(nice_name_var_dic)
tabel_dT_slcfs2 = tabel_dT_slcfs2[subset_scen_fl]
tabel_dT_anthrop2 = tabel_dT_anthrop[subset_scen_fl]

# %%

fig, axs = plt.subplots(1, len(years), figsize=[10, 4.5], sharex=False, sharey=True)
tits = ['Change in GSAT in 2040 relative to 2019', 'Change in GSAT in 2100 relative to 2019']
for yr, ax, tit in zip(years, axs, tits):
    scenario_tot = 'Scenario total'
    # Pick out year and do various renames:
    # Total antropogenic
    tot_yr = tabel_dT_anthrop2.loc[yr].rename(
        {
            'Total': scenario_tot,
            'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'
        }
    )
    # Sum SLCFs
    # Plot bars for anthropopogenic total:
    ax.barh(tot_yr.transpose().index, tot_yr.transpose()[scenario_tot].values, color='k', label='Scenario total',
            alpha=.2,
            )
    # Plot bars for SLCFs total:
    scenario_tot = 'Sum SLCFs'

    # Plot stacked plot of components:
    _tab = tabel_dT_slcfs2.loc[yr].transpose().rename({'ssp370-lowNTCF-aerchemmip': 'ssp370-lowNTCF\n-aerchemmip'})
    _tab = _tab.rename(nice_name_var_dic)
    a = _tab.plot(kind='barh', stacked=True, ax=ax, color=ls, legend=(yr != '2040'))  # , grid=True)#stac)
    _t = _tab.sum(axis=1)  # , c=100)#.plot(kind='barh', )
    ax.scatter(_t, list(_t.reset_index().index), zorder=10, c='w', marker='d')
    print(_t)
    # a = _tab.plot(kind='barh', stacked=True, ax=ax, color=ls, legend=(yr != '2040'))  # , grid=True)#stac)
    if not yr == '2040':
        ax.legend(frameon=False, ncol=1)  # [l],labels=['Sce!!nario total'], loc = 4)#'lower right')
    # Zero line:
    ax.axvline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_xlabel('$\Delta$ GSAT ($^\circ$C)')
    ax.xaxis.set_minor_locator(MultipleLocator(.1))
    # ax.grid(axis='y', which='major')

ax = axs[0]

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(right=False, left=False)  # , color='w')

ax = axs[1]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(labelleft=False, right=False, left=False, color='w')
ax.tick_params(labelright=False, labelleft=False, right=False, left=False, color='w')
ax.yaxis.set_visible(False)
plt.tight_layout()

fn = FIGURE_DIR / 'stack_bar_influence_years_all_lowNTCF_noBC.png'
plt.tight_layout()
ax = plt.gca()

plt.savefig(fn, dpi=300)
plt.savefig(fn.with_suffix('.pdf'), dpi=300)

# %%
