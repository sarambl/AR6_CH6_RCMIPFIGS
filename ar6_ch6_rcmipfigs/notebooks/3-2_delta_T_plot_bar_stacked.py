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
# ## Method: 
#

# %% [markdown]
# ## IRF:
# Using forcings from RCMIP models and the impulse response function:
# \begin{align*}
# \text{IRF}(t)=& 0.885\cdot (\frac{0.587}{4.1}\cdot exp(\frac{-t}{4.1}) + \frac{0.413}{249} \cdot exp(\frac{-t}{249}))\\
# \text{IRF}(t)= &  \sum_{i=1}^2\frac{\alpha \cdot c_i}{\tau_i}\cdot exp\big(\frac{-t}{\tau_1}\big) 
# \end{align*}
# with $\alpha = 0.885$, $c_1=0.587$, $\tau_1=4.1$, $c_2=0.413$ and $\tau_2 = 249$.

# %% Thus we can estimate the mean surface temperature change from some referance year (here 0) by using [markdown]
# the estimated ERF$_x$ for some forcing agent $x$ as follows: 

# %% [markdown]
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %% [markdown]
# # Code + figures

# %% [markdown]
# ## Imports:

# %%
import xarray as xr
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# %load_ext autoreload
# %autoreload 2

# %%
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

#PATH_DATASET = OUTPUT_DATA_DIR + '/forcing_data_rcmip_models.nc'
PATH_DT = OUTPUT_DATA_DIR + '/dT_data_rcmip_models.nc'

# %%
FIGURE_DIR = RESULTS_DIR + '/figures/'

# %%

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'

# %% [markdown]
# ### Define variables to look at:

# %%
from ar6_ch6_rcmipfigs.utils.misc_func import trans_scen2plotlabel, new_varname

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
scenarios_fl = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF-aerchemmip',  # 'ssp370-lowNTCF', Due to mistake here
                'ssp585', 'historical']
scenarios_nhist =['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF-aerchemmip',  # 'ssp370-lowNTCF', Due to mistake here
                'ssp585']# list(set(scenarios_fl)- {'historical'})
climatemodels_fl = ['Cicero-SCM', 'Cicero-SCM-ECS3', 'FaIR-1.5-DEFAULT', 'MAGICC7.1.0.beta-rcmip-phase-1', 'OSCARv3.0']

# List of delta T for variables
name_deltaT = 'Delta T'
variables_dt_comp = [new_varname(var, name_deltaT) for var in variables_erf_comp]


# %%
# Years to plot:
years = ['2040', '2100']

#scn_trans = [trans_scen2plotlabel(label) for label in scenarios_nhist]
scn_trans = scenarios_nhist # scenarios except historical

variables_tot = ['Total']
variables_sum = ['Sum SLCFs']
iterables = [years, [var.split('|')[-1] for var in variables_erf_comp]]
iterables2 = [years, variables_tot]
iterables3 = [years, variables_sum]



def setup_table(scenario_n='',
                its=None):
    if its is None:
        its = [years, [var.split('|')[-1] for var in variables_erf_comp]]

    _i = pd.MultiIndex.from_product(its, names=['', ''])
    table = pd.DataFrame(columns=scn_trans, index = _i)#.transpose()
    table.index.name=scenario_n
    return table


def setup_table_prop(scenario_n='', years=None, vars=None):
    if vars is None:
        vars = [var.split('|')[-1] for var in variables_erf_comp]
    if years is None:
        years = ['2040', '2100']
    its = [years, vars]
    _i = pd.MultiIndex.from_product(its, names=['', ''])
    table = pd.DataFrame(columns=scn_trans, index = _i)#.transpose()
    table.index.name=scenario_n
    return table


def setup_table3(scenario_n=''):
    return setup_table(scenario_n=scenario_n, its=iterables3)



setup_table()

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


# %%

# %%



# %% [markdown]
# ## Compute sum of all SLCF forcers

# %%
from ar6_ch6_rcmipfigs.utils.misc_func import get_cmap_dic, get_scenario_ls_dic, get_scenario_c_dic

# %%

# ds_DT = dic_ds[0.885]
s_y = '1850'
# cdic = get_scenario_c_dic()

cdic = get_scenario_c_dic()# get_cmap_dic(ds_DT[scenario].values)
lsdic = get_scenario_ls_dic()# _scget_ls_dic(ds_DT[climatemodel].values)


def sum_name(var): return '|'.join(var.split('|')[0:2]) + '|' + 'All'


var = variables_erf_comp[0]
f_totn = sum_name(var)
dt_totn = sum_name(new_varname(var, name_deltaT))

# make xarray with variable as new dimension:
_lst_f = []
_lst_dt = []
for var in variables_erf_comp:
    _lst_f.append(ds_DT[var])
    _lst_dt.append(ds_DT[new_varname(var, name_deltaT)])
erf_all = sum_name('Effective Radiative Forcing|Anthropogenic|all')
dt_all = sum_name(new_varname('Effective Radiative Forcing|Anthropogenic|all', name_deltaT))
ds_DT[erf_all] = xr.concat(_lst_f, pd.Index(variables_erf_comp, name='variable'))
ds_DT[dt_all] = xr.concat(_lst_dt, pd.Index(variables_erf_comp, name='variable'))

# %%
ref_year='2021'
scntab_dic = {}
#tab_tot = setup_table2()
#tab_tot_sd = setup_table2()
def table_of_sts(ds_DT, scenarios_nhist, variables, tab_vars, years, ref_year, sts = 'mean'):
    tab_tot = setup_table_prop(years=years, vars = tab_vars)
    for scn in scenarios_nhist:
        for var, tabvar in zip(variables, tab_vars):
            #tabvar = 'Total'# var.split('|')[-1]
            dtvar = new_varname(var, name_deltaT)
            tabscn = scn#trans_scen2plotlabel(scn)
            for year in years:
                _tab_da =ds_DT[dtvar].sel(scenario=scn, time=slice(year,year))-  ds_DT[dtvar].sel(scenario=scn, time=slice(ref_year,ref_year)).squeeze()
                if sts=='mean':
                    tab_tot.loc[(year, tabvar),tabscn]=_tab_da.mean('climatemodel').values[0]
                elif sts=='std':
                    tab_tot.loc[(year, tabvar),tabscn]=_tab_da.std('climatemodel').values[0]

                #tab_tot_sd.loc[(year, tabvar),tabscn]=_tab_da.std('climatemodel').values[0]
    return tab_tot
tab_tot = table_of_sts(ds_DT, scenarios_nhist, ['Delta T|Anthropogenic'], ['Total'], years, ref_year)
tab_tot_sd = table_of_sts(ds_DT, scenarios_nhist, ['Delta T|Anthropogenic'], ['Total'], years, ref_year, sts='std')

tab = table_of_sts(ds_DT, scenarios_nhist, variables_dt_comp, [var.split('|')[-1] for var in variables_dt_comp], years, ref_year)
tab_sd = table_of_sts(ds_DT, scenarios_nhist, variables_dt_comp, [var.split('|')[-1] for var in variables_dt_comp], years, ref_year, sts='std')
    #scntab_dic[scn]=tab.copy()
_ds = ds_DT.copy()
# Compute sum of SLCFs
vall = 'Delta T|Anthropogenic|All'
_ds[vall] =_ds[vall].sum('variable')
tab_slcf_sum = table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year)
tab_slcf_sum_sd = table_of_sts(_ds, scenarios_nhist, [vall], ['Sum SLCFs'], years, ref_year, sts='std')


# %%
ref_year='2021'
scntab_dic = {}
tab = setup_table()
tab_sd = setup_table()
for scn in scenarios_nhist:
    tabscn =scn# trans_scen2plotlabel(scn)

    for var in variables_erf_comp:
        tabvar = var.split('|')[-1]
        dtvar = new_varname(var, name_deltaT)
        for year in years: 
            _tab_da =ds_DT[dtvar].sel(scenario=scn, time=slice(year,year))-  ds_DT[dtvar].sel(scenario=scn, time=slice(ref_year,ref_year)).squeeze()
            #print(_tab_da)
            #print(_tab_da.mean('climatemodel').values[0])

            
            tab.loc[(year, tabvar),tabscn]=_tab_da.mean('climatemodel').values[0]
            tab_sd.loc[(year, tabvar),tabscn]=_tab_da.std('climatemodel').values[0]
    #scntab_dic[scn]=tab.copy()
tab

# %%
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,len(years), figsize=[12,8], sharey=False)
tits = ['Near Term surface temperature change (2040 relative to 2021)',
       'Long Term surface T change 2100 relatie to 2021)']
for yr, ax, tit in zip(years, axs, tits):
    tot_yr = tab_tot.loc[yr].rename({'Total':'Scenario total'})
    tot_yr.transpose().plot(kind='bar',  ax=ax, color='w', alpha=.3, edgecolor='k')#, grid=True)#stac)
    tab.loc[yr].transpose().plot(kind='bar', stacked=True, ax=ax)#, grid=True)#stac)
    ax.axhline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_ylabel('($^\circ$ C)')
fn = RESULTS_DIR+'/figures/stack_bar_influence_years.png'
plt.tight_layout()
plt.savefig(fn, dpi=300)

# %%
tot_yr = tab_tot.loc[yr].rename({'Total':ntot, 'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
tot_sd_yr = tab_tot_sd.loc[yr].rename({'Total':ntot, 'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    

plt.bar(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2, yerr=tot_sd_yr.transpose()[ntot].values)
plt.show()

# %%
sum_sd_yr.transpose()['Sum SLCFs']

# %%
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,len(years), figsize=[10,8], sharey=False)
tits = ['Near Term surface temperature change (2040 relative to 2021)',
       'Long Term surface T change 2100 relatie to 2021)']
tits = ['Change in GMST in 2040 relative to 2021', 'Change in GMST in 2100 relative to 2021']
for yr, ax, tit in zip(years, axs, tits):
    ntot = 'Scenario total'
    tot_yr = tab_tot.loc[yr].rename({'Total':ntot, 'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    tot_sd_yr = tab_tot_sd.loc[yr].rename({'Total':ntot, 'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    #l =ax.bar(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2, yerr=tab_tot_sd)
    sum_yr = tab_slcf_sum.loc[yr].rename({'Total':ntot, 'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    sum_sd_yr = tab_slcf_sum_sd.loc[yr].rename({'Total':ntot, 'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    ax.bar(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2, yerr=tot_sd_yr.transpose()[ntot].values, 
           error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ntot = 'Sum SLCFs'
    ax.bar(sum_yr.transpose().index, sum_yr.transpose()[ntot].values, color='r', label=ntot, alpha=.2, yerr=sum_sd_yr.transpose()[ntot].values,
          error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=1))

    _tab = tab.loc[yr].transpose().rename({'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    
    a = _tab.plot(kind='bar', stacked=True, ax=ax, legend=(yr!='2040'))#, grid=True)#stac)
    if not yr=='2040':
        ax.legend()#[l],labels=['Sce!!nario total'], loc = 4)#'lower right')
    
    ax.axhline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_ylabel('$\Delta$ GMST ($^\circ$C)')
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.grid(axis='y', which='major')
    
fn = RESULTS_DIR+'/figures/stack_bar_influence_years.png'
plt.tight_layout()
ax = plt.gca()

ax.tick_params(axis='y',which='minor')#,bottom='off')
plt.savefig(fn, dpi=300)

# %% [markdown]
# - De vi allerede har.
#
# Hvis vi skulle lagt til usikkerhet?
# - Ville lagt til usikkerhet gjennom ECS -- 
#     - Regne ut de samme tallene for 3 verdier av ECS. 
#     - Monte carlo trekk med en fordeling på ECS. 
#     - 

# %%
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from matplotlib.ticker import (MultipleLocator)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,len(years), figsize=[10,8], sharey=False)
tits = ['Near Term surface temperature change (2040 relative to 2021)',
       'Long Term surface T change 2100 relatie to 2021)']
tits = ['Change in GMST in 2040 relative to 2021', 'Change in GMST in 2100 relative to 2021']
for yr, ax, tit in zip(years, axs, tits):
    ntot = 'Scenario total'
    tot_yr = tab_tot.loc[yr].rename({'Total':ntot, 'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    l =ax.bar(tot_yr.transpose().index, tot_yr.transpose()[ntot].values, color='k', label='Scenario total', alpha=.2)
    _tab = tab.loc[yr].transpose().rename({'ssp370-lowNTCF-aerchemmip':'ssp370-lowNTCF\n-aerchemmip'})
    
    a = _tab.plot(kind='bar', stacked=True, ax=ax, legend=(yr!='2040'))#, grid=True)#stac)
    if not yr=='2040':
        ax.legend()#[l],labels=['Sce!!nario total'], loc = 4)#'lower right')
    
    ax.axhline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(tit)
    ax.set_ylabel('$\Delta$ GMST ($^\circ$C)')
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.grid(axis='y', which='major')
    
fn = RESULTS_DIR+'/figures/stack_bar_influence_years.png'
plt.tight_layout()
ax = plt.gca()

ax.tick_params(axis='y',which='minor')#,bottom='off')
plt.savefig(fn, dpi=300)

# %%
ax.get_axes_locator()

# %%
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,len(years), figsize=[12,8], sharey=True)
for yr, ax in zip(years, axs):
    tab_tot.loc[yr].transpose().plot(kind='bar',  ax=ax, color='k', alpha=.3)#, grid=True)#stac)
    tab.loc[yr].transpose().plot(kind='bar', stacked=True, ax=ax)#, grid=True)#stac)
    ax.axhline(0, linestyle='--', color='k', alpha=0.4)
    ax.set_title(yr+' - 2021' )
    ax.set_ylabel('($^\circ$ C)')
fn = RESULTS_DIR+'/figures/stack_bar_influence_years_same_y.png'
plt.tight_layout()
plt.savefig(fn, dpi=300)

# %%
ref_year='2021'
scntab_dic = {}
tab_to = setup_table()
for scn in scenarios_rel:
    for var in ['Delta T|Anthropogenic']:
        tabvar = var.split('|')[-1]
        dtvar = new_varname(var, name_deltaT)
        for year in years: 
            _tab_da =ds_DT[dtvar].sel(scenario=scn, time=slice(year,year))-  ds_DT[dtvar].sel(scenario=scn, time=slice(ref_year,ref_year)).squeeze()
            #print(_tab_da)
            #print(_tab_da.mean('climatemodel').values[0])
            tab_tot.loc[(scn, tabvar),year]=_tab_da.mean('climatemodel').values[0]
    #scntab_dic[scn]=tab.copy()
tab

# %%
#scntab_dic = {}
tab = setup_table2()#scenario_n=scn)

for scn in scenarios_fl:
    for var in variables_erf_comp:
        tabvar = var.split('|')[-1]
        dtvar = new_varname(var, name_deltaT)
        print(dtvar)
        for key in ECS2ecsf:
            for year in years: 
                
                _tab_da = dic_ds[key][dtvar].sel(scenario=scn, time=slice(year,year))-  dic_ds[key][dtvar].sel(scenario=scn, time=slice(ref_year,ref_year)).squeeze()
                #print(_tab_da)

                #_tab_da = dic_ds[key][var].sel(scenario=scn, time=slice(year,year))
                #print(_tab_da['climatemodel'])
                tab.loc[(scn, tabvar), (key,year)] =_tab_da.mean('climatemodel').values[0]
    #scntab_dic[scn]=tab.copy()


#tab

# %%
ds_diff = ds_DT.sel(time='2100').squeeze()-ds_DT.sel(time='2021').squeeze()

# %% [markdown]
# # Plot

# %%
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


s_y = '2021'
s_y2 = '2000'
e_y = '2100'
e_y2 = '2100'

scenarios_ss = ['ssp126','ssp245', 'ssp585']
ref_var_erf = 'Effective Radiative Forcing|Anthropogenic'
ref_var_dt = new_varname(ref_var_erf, name_deltaT)
# make subset and ref to year s_y:
_ds = ds_DT.sel(scenario=scenarios_ss, time=slice(s_y2, e_y2)) - ds_DT.sel(scenario=scenarios_ss,
                                                                           time=slice(s_y, s_y)).squeeze()
cdic1 = get_cmap_dic(variables_erf_comp, palette='bright')
cdic2 = get_cmap_dic(variables_dt_comp, palette='bright')
cdic = dict(**cdic1, **cdic2)
first=True
for ref_var, varl in zip([ref_var_dt],
                             [variables_dt_comp, variables_erf_comp]):
    fig, ax = plt.subplots(1, figsize=[7, 4.5])
    ax.plot(_ds['time'], np.zeros(len(_ds['time'])), c='k', alpha=0.5, linestyle='dashed')
    # print(ref_var)
    for scn in scenarios_ss[:]:
        # print(scn)
        # subtract year 
        _base = _ds[ref_var]  # _ds.sel(scenario=scn)
        _base = _base.sel(scenario=scn,
                          time=slice(s_y2, e_y2))  # -_base.sel(scenario=scn, time=slice(s_y, s_y)).squeeze()
        # .mean(climatemodel)
        base_keep = _base.mean(climatemodel)
        basep = _base.mean(climatemodel)
        basem = _base.mean(climatemodel)
        # print(base)
        if first:
            base_keep.plot(c='k', linewidth=2, linestyle='dashed', ax=ax, label='Scenario total ')
            first=False
        else:
            base_keep.plot(c='k', linewidth=2, linestyle='dashed', ax=ax, label='_nolegend_')
            
        scen_ds = _ds[varl] - base_keep
        test_df = scen_ds.sel(scenario=scn).mean(climatemodel).to_dataframe()
        for var in varl:
            if scn == scenarios_ss[0]:
                #label = '$\Delta$T ' + var.split('|')[-1]
                label = ' ' + var.split('|')[-1]
            else:
                label = '_nolegend_'

            _pl_da = (_ds[var].sel(scenario=scn, time=slice(s_y2, e_y2)).mean(climatemodel))  # -base_keep)
            if _pl_da.mean() <= 0:
                # print(var)

                ax.fill_between(base_keep['time'].values, basep, -_pl_da + basep, alpha=0.5,
                                color=cdic[var], label=label)
                basep = basep - _pl_da
            else:
                ax.fill_between(base_keep['time'].values, basem, basem - _pl_da, alpha=0.5,
                                color=cdic[var], label=label)
                basem = basem - _pl_da
        if 'Delta T' in ref_var:
            x_val = '2100'
            y_val = base_keep.sel(time=x_val)
            if scn == 'ssp585':
                kwargs = {'xy': (x_val, y_val )}#, 'rotation': 28.6}
            else:
                kwargs = {'xy': (x_val, y_val )}
            #ax.annotate('$\Delta$T, %s' % scn, **kwargs)
            ax.annotate(' %s' % scn, **kwargs)

    ax.legend(frameon=False, loc=2)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlim([s_y2, e_y2])
    #ax.set_ylabel('$\Delta$T (C$^\circ$)')
    ax.set_ylabel('($^\circ$C)')
    ax.set_xlabel('')
    plt.title('Temperature change contributions by SLCF\'s in two scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR +'/ssp858_126_relative_contrib.png', dpi=300)
    plt.show()

# %%
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


s_y = '2021'
s_y2 = '2000'
e_y = '2100'
e_y2 = '2100'

scenarios_ss = ['ssp126','ssp245', 'ssp585']
ref_var_erf = 'Effective Radiative Forcing|Anthropogenic'
ref_var_dt = new_varname(ref_var_erf, name_deltaT)
# make subset and ref to year s_y:
_ds = ds_DT.sel(scenario=scenarios_ss, time=slice(s_y2, e_y2)) - ds_DT.sel(scenario=scenarios_ss,
                                                                           time=slice(s_y, s_y)).squeeze()
cdic1 = get_cmap_dic(variables_erf_comp, palette='bright')
cdic2 = get_cmap_dic(variables_dt_comp, palette='bright')
cdic = dict(**cdic1, **cdic2)
first=True
for ref_var, varl in zip([ref_var_dt],
                             [variables_dt_comp, variables_erf_comp]):
    fig, ax = plt.subplots(1, figsize=[7, 4.5])
    ax.plot(_ds['time'], np.zeros(len(_ds['time'])), c='k', alpha=0.5, linestyle='dashed')
    # print(ref_var)
    for scn in scenarios_ss[:]:
        # print(scn)
        # subtract year 
        _base = _ds[ref_var]  # _ds.sel(scenario=scn)
        _base = _base.sel(scenario=scn,
                          time=slice(s_y2, e_y2))  # -_base.sel(scenario=scn, time=slice(s_y, s_y)).squeeze()
        
        base_keep = _base.mean(climatemodel)
        basep = _base.mean(climatemodel)
        basem = _base.mean(climatemodel)
        if first:
            base_keep.plot(c='k', linewidth=2, linestyle='dashed', ax=ax, label='Scenario total ')
            first=False
        else:
            base_keep.plot(c='k', linewidth=2, linestyle='dashed', ax=ax, label='_nolegend_')
            
        #scen_ds = _ds[varl] - base_keep
        #test_df = scen_ds.sel(scenario=scn).mean(climatemodel).to_dataframe()
        for var in varl:
            if scn == scenarios_ss[0]:
                #label = '$\Delta$T ' + var.split('|')[-1]
                label = ' ' + var.split('|')[-1]
            else:
                label = '_nolegend_'

            _pl_da = (_ds[var].sel(scenario=scn, time=slice(s_y2, e_y2)).mean(climatemodel))  # -base_keep)
            if _pl_da.mean() <= 0:
                # print(var)

                ax.fill_between(base_keep['time'].values, basep+_pl_da, basep, alpha=0.5,
                                color=cdic[var], label=label)
                basep = basep + _pl_da
            else:
                ax.fill_between(base_keep['time'].values, basem, basem + _pl_da, alpha=0.5,
                                color=cdic[var], label=label)
                basem = basem + _pl_da
        if 'Delta T' in ref_var:
            x_val = '2100'
            y_val = base_keep.sel(time=x_val)
            if scn == 'ssp585':
                kwargs = {'xy': (x_val, y_val ), 'rotation': 0}#28.6}
            else:
                kwargs = {'xy': (x_val, y_val )}
            #ax.annotate('$\Delta$T, %s' % scn, **kwargs)
            ax.annotate(' %s' % scn, **kwargs)

    ax.legend(frameon=False, loc=2)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlim([s_y2, e_y2])
    #ax.set_ylabel('$\Delta$T (C$^\circ$)')
    ax.set_ylabel('($^\circ$C)')
    ax.set_xlabel('')
    plt.title('Temperature change contributions by SLCF\'s in two scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR +'/ssp858_126_relative_contrib_rev.png', dpi=300)
    plt.show()

# %%
scenarios_fl


# %%
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


s_y = '2021'
s_y2 = '2000'
e_y = '2100'
e_y2 = '2100'

scenarios_ss = list(set( scenarios_fl) - {'historical','ssp370-lowNTCF-aerchemmip'})# 'ssp'#['ssp126','ssp245', 'ssp585']
ref_var_erf = 'Effective Radiative Forcing|Anthropogenic'
ref_var_dt = new_varname(ref_var_erf, name_deltaT)
# make subset and ref to year s_y:
_ds = ds_DT.sel(scenario=scenarios_ss, time=slice(s_y2, e_y2)) - ds_DT.sel(scenario=scenarios_ss,
                                                                           time=slice(s_y, s_y)).squeeze()
cdic1 = get_cmap_dic(variables_erf_comp, palette='bright')
cdic2 = get_cmap_dic(variables_dt_comp, palette='bright')
cdic = dict(**cdic1, **cdic2)
first=True
for ref_var, varl in zip([ref_var_dt],
                             [variables_dt_comp, variables_erf_comp]):
    fig, ax = plt.subplots(1, figsize=[7, 4.5])
    ax.plot(_ds['time'], np.zeros(len(_ds['time'])), c='k', alpha=0.5, linestyle='dashed')
    # print(ref_var)
    for scn in scenarios_ss[:]:
        # print(scn)
        # subtract year 
        _base = _ds[ref_var]  # _ds.sel(scenario=scn)
        _base = _base.sel(scenario=scn,
                          time=slice(s_y2, e_y2))  # -_base.sel(scenario=scn, time=slice(s_y, s_y)).squeeze()
        
        base_keep = _base.mean(climatemodel)
        basep = _base.mean(climatemodel)
        basem = _base.mean(climatemodel)
        if first:
            base_keep.plot(c='k', linewidth=2, linestyle='dashed', ax=ax, label='Scenario total ')
            first=False
        else:
            base_keep.plot(c='k', linewidth=2, linestyle='dashed', ax=ax, label='_nolegend_')
            
        #scen_ds = _ds[varl] - base_keep
        #test_df = scen_ds.sel(scenario=scn).mean(climatemodel).to_dataframe()
        for var in varl:
            if scn == scenarios_ss[0]:
                #label = '$\Delta$T ' + var.split('|')[-1]
                label = ' ' + var.split('|')[-1]
            else:
                label = '_nolegend_'

            _pl_da = (_ds[var].sel(scenario=scn, time=slice(s_y2, e_y2)).mean(climatemodel))  # -base_keep)
            if _pl_da.mean() <= 0:
                # print(var)

                ax.fill_between(base_keep['time'].values, basep+_pl_da, basep, alpha=0.5,
                                color=cdic[var], label=label)
                basep = basep + _pl_da
            else:
                ax.fill_between(base_keep['time'].values, basem, basem + _pl_da, alpha=0.5,
                                color=cdic[var], label=label)
                basem = basem + _pl_da
        if 'Delta T' in ref_var:
            x_val = '2100'
            y_val = base_keep.sel(time=x_val)
            if scn == 'ssp585':
                kwargs = {'xy': (x_val, y_val ), 'rotation': 0}#28.6}
            else:
                kwargs = {'xy': (x_val, y_val )}
            #ax.annotate('$\Delta$T, %s' % scn, **kwargs)
            ax.annotate(' %s' % scn, **kwargs)

    ax.legend(frameon=False, loc=2)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlim([s_y2, e_y2])
    #ax.set_ylabel('$\Delta$T (C$^\circ$)')
    ax.set_ylabel('($^\circ$C)')
    ax.set_xlabel('')
    plt.title('Temperature change contributions by SLCF\'s in two scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR +'/ssp858_126_relative_contrib_rev.png', dpi=300)
    plt.show()

# %%
_pl_da

# %% [markdown]
# # Alternative: Do one graph for each component AND the total? 

# %% [markdown]
# ## What question does the graph answer?
# - What are the relative contributions of SLCFs?
#     - The figure above shows the contributions of 5 SLCFs and the total anthropogenic forcing in two scenarios (black line) relative to year 2021. The area signifies the warming (below the total) or cooling (above the stipled line) introduced by changes in the SLCFer in the specific scenario. Note that in the in the businiss as usual scenario, all the SLCFers except BC on snow add to the warming, while in the 126 scenario, the emission control acts to reduce methane, ozone and BC, and these are thus contributing to cooling. Both scenarios include emission controls which act to reduce aerosols relative 2021 and thus the aerosols give warming. However, the warming from aerosols is much stronger in ssp126 because of stricter emission control in this scenario. 
#
# - 
#

# %% [markdown]
# ## What question does the graph answer?
# - How much can we gain by implementing addtional SLCF cuts?
#     - The answer to the question depends on the emission scenario we follow, because the effect of additional cuts naturally depend on how much has already been cut of a specific SLCFer. If we take scenario 119 as a baseline for the feasible cuts to SLCFs, we can calculate how much heating/cooling each component contributes with relative to this scenario. This underlines a more general point: SLCFs like ADD EXAMPLES are highly coupled to CO$_2$ emissions, which imply that strict emission control on these will automatically also control emissions of the SLCFs.  
# The figure above shows the contributions of 5 SLCFs and the total anthropogenic forcing in two scenarios. The area signifies the added/ $\Delta$T by the component. 
#

# %% [markdown]
# ## What question does the graph answer?
# - What are the relative contributions of SLCFs?
#     - The figure above shows the contributions of 5 SLCFs and the total anthropogenic forcing in two scenarios. The area signifies the additional (above the stipled line) warming and reductions (below the stipled line) by changes in the component in the specific scenario. Note that in the in the businiss as usual scenario, all components act to add to the warming, while in the 126 scenario, the emission control acts to reduce methane, ozone and BC, and these are thus contributing to cooling. 
#
