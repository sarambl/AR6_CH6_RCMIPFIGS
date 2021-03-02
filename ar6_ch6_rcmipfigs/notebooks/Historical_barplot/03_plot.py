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

# %%
import pandas as pd
import xarray as xr
from IPython.display import clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR

# %% [markdown]
# ### General about computing $\Delta T$:
# %% [markdown]
# # Code + figures

# %% [markdown]
# ### Path input data

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR, BASE_DIR

#PATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'
PATH_DATASET = OUTPUT_DATA_DIR/'historic_delta_GSAT/dT_data_hist_recommendation.nc'









# %% [markdown]
# ## Path output data

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
#PATH_DT_TAB_OUTPUT = RESULTS_DIR / 'tables' / 'table_sens_dT_cs_recommandetion.csv'
PATH_DF_OUTPUT = OUTPUT_DATA_DIR / 'historic_delta_GSAT/dT_data_hist_recommendation.csv'

print(PATH_DF_OUTPUT)


# %% [markdown]
# ## various definitions

# %% [markdown]
# Year to integrate from and to:

# %%
first_y = 1750
last_y = 2019

# %% [markdown]
# **Set reference year for temperature change:**

# %%
ref_year = 1750

# %% [markdown]
# ### Define variables to look at:

# %%
# variables to plot:
variables_erf_comp = [
    'CO2', 'N2O', 'CH4', 'HC', 'NOx', 'SO2', 'BC', 'OC', 'NH3','VOC'
]
# total ERFs for anthropogenic and total:
variables_erf_tot = []
variables_all = variables_erf_comp + variables_erf_tot
# Scenarios to plot:
scenarios_fl = []

# %% [markdown]
# ### Open ERF dataset:

# %%
ds = xr.open_dataset(PATH_DATASET)
ds['Delta T']

# %%
ds['variable']

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_cmap_dic

# %%
fig, axs = plt.subplots(2, sharex=True, figsize=[6,6])

ax_erf = axs[0]
ax_dT = axs[1]
for v in ds['variable'].values:
    ds.sel(variable=v)['Delta T'].plot(ax=ax_dT, label=v, c=cols[v])
    ds.sel(variable=v)['ERF'].plot(ax=ax_erf, c=cols[v])
ds.sum('variable')['Delta T'].plot(ax=ax_dT, label='Sum', c='k',linewidth=2)
ds.sum('variable')['ERF'].plot(ax=ax_erf, c='k',linewidth=2)
    
ax_dT.set_title('Temperature change')
ax_erf.set_title('ERF')
ax_erf.set_ylabel('ERF [W m$^{-2}$]')
ax_dT.set_ylabel('$\Delta$ GSAT [$^{\circ}$C]')
ax_erf.set_xlabel('')
ax_dT.legend(ncol=4, loc='upper left', frameon=False)
plt.tight_layout()
fig.savefig('hist_timeseries_ERF_dT.png', dpi=300)

# %%
fig, axs = plt.subplots(2, sharex=True, figsize=[6,6])

ax_erf = axs[0]
ax_dT = axs[1]
for v in ds['variable'].values:
    ds.sel(variable=v)['Delta T'].plot(ax=ax_dT, label=v, c=cols[v])
    ds.sel(variable=v)['ERF'].plot(ax=ax_erf, c=cols[v])
ax_dT.set_title('Temperature change')
ax_erf.set_title('ERF')
ax_erf.set_ylabel('ERF [W m$^{-2}$]')
ax_dT.set_ylabel('$\Delta$ GSAT [$^{\circ}$C]')
ax_erf.set_xlabel('')
ax_dT.legend(ncol=4, loc='upper left', frameon=False)
plt.tight_layout()

# %%
df_deltaT = ds['Delta T'].squeeze().drop('percentile').to_dataframe().unstack('variable')['Delta T']
fig, ax = plt.subplots(figsize=[10,5])
for v in variables_all:
    df_deltaT[variables_all][v].plot(linewidth=3,ax = ax, label=v, color=cols[v])#, color=cols.items())
plt.legend(loc='upper left')
plt.ylabel('$\Delta$ T ($^\circ$ C)')

# %%
col_list = [cols[c] for c in df_deltaT.columns]
col_list


# %%
import seaborn as sns

# %%
df_deltaT = ds['Delta T'].squeeze().drop('percentile').to_dataframe().unstack('variable')['Delta T']

fig, ax = plt.subplots(figsize=[10,5])
ax.hlines(0,1740,2028, linestyle='solid',alpha=0.9, color='k', linewidth=0.5)#.sum(axis=1).plot(linestyle='dashed', color='k', linewidth=3)

df_deltaT.plot.area( color=col_list, ax=ax)
df_deltaT.sum(axis=1).plot(linestyle='dashed', color='k', linewidth=3, label='Sum')
plt.legend(loc='upper left',ncol=3, frameon=False)
plt.ylabel('$\Delta$ GSAT ($^\circ$ C)')
ax.set_xlim([1740,2028])
sns.despine()

# %%
import seaborn as sns


# %%
fig, ax = plt.subplots()
ax.vlines(0,-1,3, linestyle='dashed',alpha=0.4)
yrs = [1950,1960,1970,1980,1990,2000, 2019,]
labs ={y:f'{y}-1750' for y in yrs}
df_deltaT.loc[yrs].rename(labs,axis=0).plot.barh(stacked=True, color=col_list, ax=ax)
plt.legend(frameon=False)
sns.despine(fig, left=True)
ax.set_xlim([-1,2.3])
ax.set_xlabel('$\Delta$GSAT$^\circ$C')
ax.set_ylabel('')
plt.show()

# %% [markdown]
# ## Split up into components

# %% [markdown]
# We use the original split up from bill collins plot 

# %%
from ar6_ch6_rcmipfigs.constants import BASE_DIR
# file path table of ERF 2019-1750
fp_collins = BASE_DIR/'notebooks/bill_collins/table_mean_smb_orignames.csv'
fp_collins_sd = BASE_DIR/'notebooks/bill_collins/table_std_smb_orignames.csv'

# %%
fp_collins

# %%
import pandas as pd

# %%
df_collins = pd.read_csv(fp_collins, index_col=0)
df_collins.index = df_collins.index.rename('emission_experiment')
df_collins_sd = pd.read_csv(fp_collins_sd, index_col=0)
df_collins

# %%
varn = ['co2','N2O','HC','WMGHG','ch4','o3','H2O_strat','ari','aci']
var_dir = ['CO2','N2O','HC','GHG','CH4_lifetime','O3','Strat_H2O','Aerosol','Cloud']

# %%
rename_dic_cat = {
    'CO2':'Carbon dioxide (CO$_2$)',
    'GHG':'WMGHG',
    'CH4_lifetime': 'Methane (CH$_4$)',
    'O3': 'Ozone (O$_3$)',
    'Strat_H2O':'H$_2$O (strat)',
    'Aerosol':'Aerosol-radiation',
    'Cloud':'Aerosol-cloud',
    'N2O':'N$_2$O',
    'HC':'CFC + HCFC',

}
rename_dic_cols ={
    'CO2':'CO$_2$',
    'CH4':'CH$_4$',
    'N2O':'N$_2$O',
    'HC':'CFC + HCFC',
    'NOx':'NO$_x$',
    'VOC':'NMVOC + CO',
    'SO2':'SO$_2$',
    'OC':'Organic carbon',
    'BC':'Black carbon',
    'NH3':'Ammonia'
}
tab_plt_ERF = df_collins.loc[::-1,var_dir].rename(rename_dic_cat, axis=1).rename(rename_dic_cols, axis=0)#.transpose()
tab_plt = tab_plt_ERF
tab_plt_ERF


# %%
import pandas as pd
num_mod_lab = 'Number of models (Thornhill 2020)'
thornhill = pd.read_csv('../bill_collins/table2_thornhill2020.csv', index_col=0)
thornhill.index = thornhill.index.rename('Species')
thornhill

std_2_95th = 1.645

sd_tot = df_collins_sd['Total_sd']
df_err= pd.DataFrame(sd_tot.rename('std'))
df_err['SE'] = df_err

df_err['SE'] = df_err['std']/np.sqrt(thornhill[num_mod_lab])
df_err['95-50_SE'] = df_err['SE']*std_2_95th
df_err.loc['CO2','95-50_SE']= df_err.loc['CO2','std']
df_err

df_err['95-50'] = df_err['std']*std_2_95th
df_err.loc['CO2','95-50']= df_err.loc['CO2','std']
df_err

# %%
df_err = df_err.rename(rename_dic_cols, axis=0)

# %%
width = 0.7
kwargs = {'linewidth':.1,'edgecolor':'k'}


# %%
import seaborn as sns

# %%
ybar = np.arange(len(tab_plt)+1)#, -1)
ybar

# %%
col

# %%
fig, ax = plt.subplots()#figsize=[10,10])
cmap = get_cmap_dic(var_dir)
col_ls = [cmap[v] for v in var_dir]
tab_plt.plot.barh(stacked=True, color=col_ls, linewidth=.0, edgecolor='k',ax=ax, width=width)
#tot = table['Total'][::-1]
tot = tab_plt.sum(axis=1)#tab_plt
xerr = df_err['95-50'][::-1]
y = np.arange(len(tot))
plt.errorbar(tot, y,xerr=xerr,marker='d', linestyle='None', color='k', label='Sum', )
plt.legend(frameon=False)
ax.set_ylabel('')
sns.despine()





for lab, y in zip(tab_plt.index, ybar):
        #plt.text(-1.55, ybar[i], species[i],  ha='left')#, va='left')
    plt.text(-1.9, y-0.1, lab,  ha='left')#, va='left')
plt.title('Change in effective radiative forcing from  1750 to 2019')
plt.xlabel(r'Effective radiative forcing, W m$^{-2}$')
plt.xlim(-1.5, 2.6)
    #plt.xlim(-1.6, 2.0)
sns.despine(fig, left=True, trim=True)
plt.legend(loc='lower right', frameon=False)
plt.axvline(x=0., color='k', linewidth=0.25)
fn = 'attribution_1750_2019_5-95th.png'
fp = RESULTS_DIR /'figures_historic_attribution'/fn
fp.parent.mkdir(parents=True, exist_ok=True)
ax.set_yticks([])

#plt.savefig(fp, dpi=300)
#plt.savefig(fp.with_suffix('.pdf'), dpi=300)
plt.show()


# %% [markdown]
# ## Scale delta GSAT by ERF 

# %%
df_collins#.columns

# %% [markdown]
# ## Source of delta T equal to source of ERF

# %%
dT_2019 = pd.DataFrame(df_deltaT.loc[2019])
dT_2019.index= dT_2019.index.rename('emission_experiment')
df_coll_t = df_collins.transpose()
if 'Total' in df_coll_t.index:
    df_coll_t = df_coll_t.drop('Total')
# scale by total:
scale = df_coll_t.sum()
# normalized ERF: 
df_col_normalized = df_coll_t/scale
#
df_col_normalized.transpose().plot.barh(stacked=True)

# %% [markdown]
# We multiply the change in GSAT in 2019 by the matrix describing the source distribution from the ERF:

# %%
df_dt_sep = dT_2019[2019]*df_col_normalized
df_dt_sep=df_dt_sep.transpose()

# %%
df_dt_sep.plot.bar(stacked=True)

# %%
df_dt_sep.transpose().sum().plot.line()

df_deltaT.loc[2019].plot.line()#bar(stacked=True)

# %%
rename_dic_cat = {
    'CO2':'Carbon dioxide (CO$_2$)',
    'GHG':'WMGHG',
    'CH4_lifetime': 'Methane (CH$_4$)',
    'O3': 'Ozone (O$_3$)',
    'Strat_H2O':'H$_2$O (strat)',
    'Aerosol':'Aerosol-radiation',
    'Cloud':'Aerosol-cloud'
}
rename_dic_cols ={
    'CO2':'CO$_2$',
    'CH4':'CH$_4$',
    'N2O':'N$_2$O',
    'HC':'CFC + HCFC',
    'NOx':'NO$_x$',
    'VOC':'VOC + CO',
    'SO2':'SO$_2$',
    'OC':'Organic carbon',
    'BC':'Black carbon',
    'NH3':'Ammonia'
}


# %%
ybar = np.arange(len(tab_plt)+1)#, -1)
ybar

# %%
col_ls = [cols[c] for c in cols.keys()]

# %%
exps_ls = ['CO2', 'CH4', 'N2O', 'HC', 'NOx', 'VOC', 'SO2', 'OC', 'BC', 'NH3']

# %%
tab_plt_dT = df_dt_sep.loc[::-1,var_dir]#.rename(rename_dic_cat, axis=1).rename(rename_dic_cols, axis=0)
tab_plt_dT=tab_plt_dT.loc[exps_ls]
tab_plt_dT =tab_plt_dT.rename(rename_dic_cat, axis=1).rename(rename_dic_cols, axis=0)

# %%
cmap = get_cmap_dic(tab_plt.columns)
col_ls = [cmap[c] for c in cmap.keys()]

# %%
tab_plt.sum(axis=1)[::-1]

# %%
fig, axs = plt.subplots(1,2,dpi=150, figsize=[10,4])
width=.8
kws = {
    'width':.8,
    'linewidth':.1,
    'edgecolor':'k',
    
}

ax=axs[0]
tab_plt_ERF.plot.barh(stacked=True, color=col_ls, ax=ax,**kws)
#tot = table['Total'][::-1]
tot = tab_plt_ERF.sum(axis=1)#tab_plt
xerr = df_err['95-50'][::-1]
y = np.arange(len(tot))
ax.errorbar(tot, y,xerr=xerr,marker='d', linestyle='None', color='k', label='Sum', )
#ax.legend(frameon=False)
ax.set_ylabel('')


for lab, y in zip(tab_plt.index, ybar):
        #plt.text(-1.55, ybar[i], species[i],  ha='left')#, va='left')
    ax.text(-1.9, y-0.1, lab,  ha='left')#, va='left')
ax.set_title('Effective radiative forcing, 1750 to 2019')
ax.set_xlabel(r'Effective radiative forcing, W m$^{-2}$')
#ax.set_xlim(-1.5, 2.6)
    #plt.xlim(-1.6, 2.0)
#sns.despine(fig, left=True, trim=True)
ax.legend(loc='lower right', frameon=False)
ax.axvline(x=0., color='k', linewidth=0.25)
ax.set_yticks([])

ax.get_legend().remove()





ax=axs[1]
tab_plt_dT[::-1].plot.barh(stacked=True, color=col_ls, ax=ax,**kws)
tot = tab_plt_dT.sum(axis=1)[::-1]
#xerr =0# df_err['95-50'][::-1]
y = np.arange(len(tot))
ax.errorbar(tot, y,marker='d', linestyle='None', color='k', label='Sum', )
#ax.legend(frameon=False)
ax.set_ylabel('')





##for lab, y in zip(tab_plt_dT[::-1].index, ybar):
        #plt.text(-1.55, ybar[i], species[i],  ha='left')#, va='left')
#    ax.text(-.8, y-0.1, lab,  ha='left')#, va='left')
ax.set_title('Change in GSAT,  1750 to 2019')
ax.set_xlabel(r'Change in GSAT ($^{\circ}$C)')
ax.set_xlim(-.6, 1.8)


sns.despine(fig, left=True, trim=True)
ax.spines['bottom'].set_bounds(-.5,1.5)

ax.legend(loc='lower right', frameon=False)
ax.axvline(x=0., color='k', linewidth=0.25)
fn = 'attribution_1750_2019.png'
#fp = RESULTS_DIR /'figures_historic_attribution_DT'/fn
#fp.parent.mkdir(parents=True, exist_ok=True)
ax.set_yticks([])

#plt.savefig(fp, dpi=300)
#plt.savefig(fp.with_suffix('.pdf'), dpi=300)
plt.show()


# %%
(dT_2019[2019]*df_col_normalized).transpose().plot.barh(stacked=True)

# %%
dT_2019[2019].plot.barh(stacked=True)

# %%
scale = np.abs(df_coll_t.drop('Total')).sum()#).transpose().plot.barh(stacked=True)

# %%
np.abs((df_coll_t.drop('Total')/np.abs(df_coll_t.drop('Total')).sum()).transpose()).sum(axis=1)#.plot.barh(stacked=True)

# %%
forcing_total_collins = df_collins['Total']

# %%
sd_tot = table_sd['Total_sd']
df_err= pd.DataFrame(sd_tot.rename('std'))
df_err['SE'] = df_err

# %%

std_2_95th = 1.645#df_err['SE'] = df_err['std']/np.sqrt(thornhill[num_mod_lab])
df_err['95-50_SE'] = df_err['SE']*std_2_95th
df_err.loc['CO2','95-50_SE']= df_err.loc['CO2','std']


# %%
df_err['95-50'] = df_err['std']*std_2_95th
df_err.loc['CO2','95-50']= df_err.loc['CO2','std']
df_err

# %%
std_2_95th = 1.645

# %%
var_dir

# %%
cmap = get_cmap_dic(var_dir)

cmap

cols = [cmap[c] for c in var_dir]

# %%

# %%
ybar = np.arange(len(tab_plt)+1)#, -1)
ybar

# %%
tab_plt= pd.read_csv(BASE_DIR/'notebooks/bill_collins/plt_tab.csv',index_col=0)
tot=pd.read_csv(BASE_DIR/'notebooks/bill_collins/tot.csv',index_col=0)
xerr= pd.read_csv(BASE_DIR/'notebooks/bill_collins/xerr.csv',index_col=0)

# %%
width = 0.7
kwargs = {'linewidth':.1,'edgecolor':'k'}

# %%
fig, ax = plt.subplots()#figsize=[10,10])


tab_plt.plot.barh(stacked=True, color=cols, linewidth=.0, edgecolor='k',ax=ax, width=width)
#tot = table['Total']#[::-1]
#xerr = df_err['95-50']#[::-1]
y = np.arange(len(tot))
plt.errorbar(tot['Total'], y,xerr=xerr['95-50'],marker='d', linestyle='None', color='k', label='Sum', )
plt.legend(frameon=False)
ax.set_ylabel('')
sns.despine()





for lab, y in zip(tab_plt.index, ybar):
        #plt.text(-1.55, ybar[i], species[i],  ha='left')#, va='left')
    plt.text(-1.9, y-0.1, lab,  ha='left')#, va='left')
plt.title('Change in effective radiative forcing from  1750 to 2019')
plt.xlabel(r'Effective radiative forcing, W m$^{-2}$')
plt.xlim(-1.5, 2.6)
    #plt.xlim(-1.6, 2.0)
sns.despine(fig, left=True, trim=True)
plt.legend(loc='lower right', frameon=False)
plt.axvline(x=0., color='k', linewidth=0.25)
fn = 'attribution_1750_2019_5-95th.png'
fp = RESULTS_DIR /'figures_historic_attribution'/fn
fp.parent.mkdir(parents=True, exist_ok=True)
ax.set_yticks([])

#plt.savefig(fp, dpi=300)
#plt.savefig(fp.with_suffix('.pdf'), dpi=300)
plt.show()


# %%
df_deltaT.loc[yrs].rename(labs,axis=0).plot.barh(stacked=True, color=col_list, ax=ax)


# %%
fig, (ax,ax2) = plt.subplots(2,  gridspec_kw={'height_ratios': [3, 1]},figsize=[10,10])


tab_plt.plot.barh(stacked=True, color=cols, linewidth=.0, edgecolor='k',ax=ax, width=width)
#tot = table['Total']#[::-1]
#xerr = df_err['95-50']#[::-1]
y = np.arange(len(tot))
ax.errorbar(tot['Total'], y,xerr=xerr['95-50'],marker='d', linestyle='None', color='k', label='Sum', )
ax.legend(frameon=False, loc='lower right')
ax.set_ylabel('')
sns.despine(fig)





for lab, y in zip(tab_plt.index, ybar):
        #plt.text(-1.55, ybar[i], species[i],  ha='left')#, va='left')
    ax.text(-1.9, y-0.1, lab,  ha='left')#, va='left')
plt.title('Change in effective radiative forcing from  1750 to 2019')
plt.xlabel(r'Effective radiative forcing, W m$^{-2}$')
plt.xlim(-1.5, 2.6)
    #plt.xlim(-1.6, 2.0)
sns.despine(fig, left=True, trim=True)
plt.legend(loc='lower right', frameon=False)
plt.axvline(x=0., color='k', linewidth=0.25)
fn = 'attribution_1750_2019_5-95th.png'
fp = RESULTS_DIR /'figures_historic_attribution'/fn
fp.parent.mkdir(parents=True, exist_ok=True)
ax.set_yticks([])

#plt.savefig(fp, dpi=300)
#plt.savefig(fp.with_suffix('.pdf'), dpi=300)







# %%
fig, ax = plt.subplots()
ax.vlines(0,-1,3, linestyle='dashed',alpha=0.4)
yrs = [1950,1960,1970,1980,1990,2000, 2019,]
labs ={y:f'{y}-1750' for y in yrs}
df_deltaT.loc[yrs].rename(labs,axis=0).plot.barh(stacked=True, color=col_list, ax=ax)
plt.legend(frameon=False)
sns.despine(fig, left=True)
ax.set_xlim([-1,2.3])
ax.set_xlabel('$\Delta$GSAT$^\circ$C')
ax.set_ylabel('')
plt.show()

# %%
type(xerr)

# %% [markdown]
# ## Meeting notes:
#
#
# - Replace WMGHG by N2O /halogenated (posibilitity to split in blue/darkblue??)
# - VOC should be NMVOC
# - Title “Emission based ERF in 2019 relative to 1750”
# - Add HFCs
# Idealized delta GSAT:
# - Mock up for zoom out from the bigger one
# - Add the emissions protocol in a corner,
# - Title is it “mitigation” and not idealized sustained reductions?
#
#
#
# ## GSAT timeseries
# - _Giddens and aerc_
# - _impact --> effect_
#
# - _Replace low NTCF by low SLCF low CH4 or low SLCF high CH4_
# - _Giddens: low CH4_
# - _aerchemip: high CH4_
# - Impact SHOULD BE effect!!
# - Add above the uncertainty bar what it represent (on the figure)
# - Add °C for the top 2 left panels
#
# - Add a second y-axis for absolute change comp w 1850
#
#
# - _relative to 2019 
#
# - Add relative to 1750
# - 
#
# ## SSP bar plot:
#
#
# - _Replace low NTCF by low SLCF low CH4 or low SLCF high CH4_
# - _Replace delta by GSAT change_
# - _relative to 2019_
