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

# %%
import seaborn as sns

# %% [markdown]
# ### General about computing $\Delta T$:
# %% [markdown]
# # Code + figures

# %%
output_name = 'fig_em_based_ERF_GSAT'

# %% [markdown]
# ### Path input data

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR, BASE_DIR

#PATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'
PATH_DATASET = OUTPUT_DATA_DIR/'historic_delta_GSAT/dT_data_hist_recommendation.nc'






fn_erf_decomposition = OUTPUT_DATA_DIR / 'historic_delta_GSAT/hist_ERF_est_decomp.csv'
fn_ERF_2019= OUTPUT_DATA_DIR/'historic_delta_GSAT/2019_ERF_est.csv'
#fn_output_decomposition = OUTPUT_DATA_DIR / 'historic_delta_GSAT/hist_ERF_est_decomp.csv'


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
cols = get_cmap_dic(ds['variable'].values)

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
m1850_1900 = df_deltaT.loc[1850:1900].mean()
m2010_2019 = df_deltaT.loc[2010:2019].mean()
diff = m2010_2019-m1850_1900
diff

# %%
m2010_2019

# %%
df_deltaT.loc[2019]

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
# # Split up into components

# %% [markdown]
# We use the original split up in ERF from Thornhill/Bill Collin's plot 

# %%
from ar6_ch6_rcmipfigs.constants import BASE_DIR, OUTPUT_DATA_DIR
# file path table of ERF 2019-1750

fp_collins =RESULTS_DIR/'tables_historic_attribution/table_mean_smb_orignames.csv'
fp_collins_sd = RESULTS_DIR/'tables_historic_attribution/table_std_smb_orignames.csv'

# %%
fn_est_ERF = OUTPUT_DATA_DIR/'historic_delta_GSAT/hist_ERF_est.csv'

# %%
import pandas as pd

# %%
fn_erf_decomposition = OUTPUT_DATA_DIR / 'historic_delta_GSAT/hist_ERF_est_decomp.csv'
fn_ERF_2019= OUTPUT_DATA_DIR/'historic_delta_GSAT/2019_ERF_est.csv'


# %%
df_collins = pd.read_csv(fn_ERF_2019, index_col=0)
df_collins.index = df_collins.index.rename('emission_experiment')
df_collins_sd = pd.read_csv(fp_collins_sd, index_col=0)
df_collins

# %% [markdown]
# ## don't add HFCs together with HC

# %%
#hfcs = df_collins['HFCs']
#df_collins = df_collins.drop('HFCs', axis=1)
#df_collins['HC'] = df_collins['HC'] + hfcs
#df_collins

# %%
varn = ['co2','N2O','HC','HFCs','ch4','o3','H2O_strat','ari','aci']
var_dir = ['CO2','N2O','HC','HFCs','CH4_lifetime','O3','Strat_H2O','Aerosol','Cloud']

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
    'HFCs':'HFC'

}
rename_dic_cols ={
    'co2':'CO$_2$',
    'CO2':'CO$_2$',
    'CH4':'CH$_4$',
    'ch4':'CH$_4$',
    'N2O':'N$_2$O',
    'n2o':'N$_2$O',
    'HC':'CFC + HCFC + HFC',
    'HFCs':'HFC',
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
thornhill = pd.read_csv(INPUT_DATA_DIR/'table2_thornhill2020.csv', index_col=0)
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
ybar = np.arange(len(tab_plt)+1)#, -1)
ybar

# %% [markdown]
# ## decompose GSAT as ERF 

# %% [markdown]
# ### Source of delta T equal to source of ERF

# %%
df_collins

# %%
df_deltaT['CO2']

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

# %% [markdown]
# #### Double check that the sum is the same as before:

# %%
df_dt_sep.transpose().sum().plot.line()

df_deltaT.loc[2019].plot.line()#bar(stacked=True)

# %% [markdown]
# ## REMOVED: For consistency with rest of the report (chapter 7), GSAT is scaled by  1.3/1.48527635

# %% [markdown]
# df_dt_sep

# %% [markdown]
# df_dt_sep.sum(axis=1)

# %% [markdown]
# dt_tmp = df_dt_sep.sum(axis=1)# + df_dt_sep
#
# ghg = dt_tmp['CO2']+dt_tmp['CH4']+dt_tmp['N2O'] + dt_tmp['NOx'] +  dt_tmp['VOC'] + dt_tmp['HC']
# BC_nosnow = 0.077972559347102
#
# aer = dt_tmp['SO2']+ dt_tmp['OC'] + dt_tmp['NH3'] + dt_tmp['BC']
# sum_ghg_aer = ghg + aer
# sum_ghg_aer
# scale_by = 1.3/sum_ghg_aer

# %% [markdown]
# print('GHG:',ghg)
# print('AER:',aer)
# print('SUM: ',sum_ghg_aer)
# print('scale by', scale_by)

# %% [markdown]
# df_dt_sep = df_dt_sep*scale_by

# %% [markdown]
# Correct order of variables:

# %% [markdown]
# ## Accounting for non-linearities in ERFaci, we scale down the ERF aci contribution to fit with chapter 7 

# %%
scal_to = -0.38
aci_tot = df_dt_sep.sum()['Cloud']
scale_by = scal_to/aci_tot
print(scal_to, aci_tot)

df_dt_sep['Cloud'] = df_dt_sep['Cloud']*scale_by
df_dt_sep.sum()

# %%
exps_ls = ['CO2', 'CH4', 'N2O', 'HC', 'NOx', 'VOC', 'SO2', 'OC', 'BC', 'NH3']

# %%
tab_plt_dT = df_dt_sep.loc[::-1,var_dir]#.rename(rename_dic_cat, axis=1).rename(rename_dic_cols, axis=0)
tab_plt_dT=tab_plt_dT.loc[exps_ls]
tab_plt_dT =tab_plt_dT.rename(rename_dic_cat, axis=1).rename(rename_dic_cols, axis=0)

# %%
cmap = get_cmap_dic(var_dir)
col_ls = [cmap[c] for c in cmap.keys()]

# %%
tab_plt_dT.sum(axis=1)

# %% [markdown]
# ### Uncertainties $\Delta$ GSAT

# %%
std_ERF =df_err['std']
std_ECS_lw_rl = 0.5/3
std_ECS_hg_rl = 1/3

tot_ERF = tab_plt_ERF.sum(axis=1)
std_erf_rl = np.abs(std_ERF/tot_ERF)
std_erf_rl


# %%
std_ERF

# %%
tot_ERF


# %% [markdown]
#
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %% [markdown]
# most of the uncertainty in the IRF derives from the uncertainty in the climate sensitivity which is said 3 (2.5-4), i.e. relative std 0.5/3 for the lower and 1/3 for the higher. If we treat this as two independent normally distributed variables multiplied together, $X$ and $Y$ and $X \cdot Y$, we may propagate the uncertainty: 
#
# \begin{align*} 
# \frac{\sigma_{XY}^2}{(XY)^2} = \Big[(\frac{\sigma_X}{X})^2 + (\frac{\sigma_Y}{Y})^2 \Big]
# \end{align*}

# %%
def rel_sigma_prod(rel_sigmaX,rel_sigmaY):
    var_prod_rel =( (rel_sigmaX)**2 + (rel_sigmaY)**2)
    rel_sigma_prod = np.sqrt(var_prod_rel)
    return rel_sigma_prod

rel_sig_lw =  rel_sigma_prod(std_erf_rl, std_ECS_lw_rl)
rel_sig_hg =  rel_sigma_prod(std_erf_rl, std_ECS_hg_rl)

# %%
tot_dT = tab_plt_dT.sum(axis=1)[::-1]

neg_v =(tot_dT<0)#.squeeze()


# %%
std_2_95th

# %%
tot_dT = tab_plt_dT.sum(axis=1)
err_dT = pd.DataFrame(index=tot_dT.index)
err_dT['min 1 sigma'] = np.abs(tot_dT*rel_sig_lw)#*tot_dT
err_dT['plus 1 sigma'] =np.abs(tot_dT*rel_sig_hg)
err_dT['plus 1 sigma'][neg_v]=np.abs(tot_dT*rel_sig_lw)[neg_v]#.iloc[neg_v].iloc[neg_v].iloc[neg_v]
err_dT['min 1 sigma'][neg_v]=np.abs(tot_dT*rel_sig_hg)[neg_v]#.iloc[neg_v].iloc[neg_v].iloc[neg_v]
#err_dT['min 1 sigma'].iloc[neg_v] =np.abs(tot_dT*rel_sig_hg).iloc[neg_v]
#err_dT['plus 1 sigma'][neg_v] = np.abs(tot_dT*rel_sig_lw)[neg_v]
#err_dT['min 1 sigma'][neg_v] = np.abs(tot_dT*rel_sig_hg)[neg_v]
#[::-1]
err_dT['p50-05'] = err_dT['min 1 sigma']*std_2_95th
err_dT['p95-50'] = err_dT['plus 1 sigma']*std_2_95th
err_dT

#var_nn_dir = [rename_dic_cols[v] for v in varn]

# %%
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


# %%
sns.set_style()
fig, axs = plt.subplots(1,2,dpi=300, figsize=[10,4])#, dpi=150)
width=.8
kws = {
    'width':.8,
    'linewidth':.1,
    'edgecolor':'k',
    
}

ax=axs[0]
ax.axvline(x=0., color='k', linewidth=0.25)

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
ax.set_xlabel(r'(W m$^{-2}$)')
#ax.set_xlim(-1.5, 2.6)
    #plt.xlim(-1.6, 2.0)
#sns.despine(fig, left=True, trim=True)
ax.legend(loc='lower right', frameon=False)
ax.set_yticks([])

ax.get_legend().remove()

ax.set_xticks(np.arange(-1.5,2.1,.5))
ax.set_xticks(np.arange(-1.5,2,.1), minor=True)




ax=axs[1]
ax.axvline(x=0., color='k', linewidth=0.25)

tab_plt_dT[::-1].plot.barh(stacked=True, color=col_ls, ax=ax,**kws)
tot = tab_plt_dT.sum(axis=1)[::-1]
#xerr =0# df_err['95-50'][::-1]
y = np.arange(len(tot))

ax.errorbar(tot, y,
            xerr=err_dT[['p50-05','p95-50']].loc[tot.index].transpose().values,
            #xerr=err_dT[['min 1 sigma','plus 1 sigma']].loc[tot.index].transpose().values,
            marker='d', linestyle='None', color='k', label='Sum', )
#ax.legend(frameon=False)
ax.set_ylabel('')

ax.set_title('Change in GSAT, 1750 to 2019')
ax.set_xlabel(r'($^{\circ}$C)')
ax.set_xlim(-1.3, 1.8)


sns.despine(fig, left=True, trim=True)
ax.spines['bottom'].set_bounds(-1.,1.5)
ax.legend(loc='lower right', frameon=False)


ax.set_xticks(np.arange(-1,2.1,.5))
    #ax.xaxis.set_major_locator(MultipleLocator(.5))
    
ax.set_xticks(np.arange(-1,1.6,.5))
ax.set_xticks(np.arange(-1,1.5,.1), minor=True)


fn = output_name + '.png'
fp = RESULTS_DIR /'figures_historic_attribution_DT'/fn
fp.parent.mkdir(parents=True, exist_ok=True)
ax.set_yticks([])
fig.tight_layout()
plt.savefig(fp, dpi=300, bbox_inches='tight')
plt.savefig(fp.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(fp.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.show()


# %%
tab_plt_ERF.sum(axis=0)

# %%
tab_plt_dT.sum(axis=1)

# %%
tab_plt_dT.sum()

# %% [markdown]
# ### Write vales to csv

# %%
fn = output_name+'_values_ERF.csv'
fp = RESULTS_DIR /'figures_historic_attribution_DT'/fn
tab_plt_ERF.to_csv(fp)


fn = output_name+'_values_ERF_uncertainty.csv'
fp = RESULTS_DIR /'figures_historic_attribution_DT'/fn
df_err.to_csv(fp)


# %%
fn = output_name+'_values_dT.csv'
fp = RESULTS_DIR /'figures_historic_attribution_DT'/fn
tab_plt_dT.to_csv(fp)


fn = output_name+'_values_dT_uncertainty.csv'
fp = RESULTS_DIR /'figures_historic_attribution_DT'/fn
err_dT.to_csv(fp)


# %%

# %%

# %%

# %%
