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
# ##

# %%
import pandas as pd
import numpy.testing
from numpy.testing import assert_allclose
from ar6_ch6_rcmipfigs.utils.plot import get_chem_col
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR
from pathlib import  Path
import numpy as np
import matplotlib.pyplot as plt


# %%
import attribution_1750_2019_v2_smb

# %%
table, table_sd = attribution_1750_2019_v2_smb.main()

# %%
table.to_csv('table_mean_smb_orignames.csv')

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_chem_col

# %%
varn = ['co2','WMGHG','ch4','o3','H2O_strat','ari','aci']
var_dir = ['CO2','GHG','CH4_lifetime','O3','Strat_H2O','Aerosol','Cloud']

# %%
cols = [get_chem_col(var) for var in varn]

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
o3_sum = table['O3']+table['O3_prime']
tab2 = table.copy(deep=True).drop(['O3','O3_prime','Total'], axis=1)
tab2['O3'] = o3_sum

# %%
tab2.loc[::-1,var_dir].plot.barh(stacked=True, color=cols, linewidth=.3, edgecolor='k',
                                )
tot = table['Total'][::-1]
xerr = table_sd['Total_sd'][::-1]
y = np.arange(len(tot))
plt.errorbar(tot, y,xerr=xerr,marker='d', linestyle='None', color='k', label='Sum', )
plt.legend()

# %% [markdown]
# ## Uncertainty:

# %% [markdown]
# We have the standard deviation, but would like the use the standard error of the mean AND we would like to calculate the 5-95th percentile. 

# %% [markdown]
# We have the standard deviation (as far as I can tell, not the unbiased one)

# %% [markdown]
# $\sigma=\sqrt {\frac {\sum _{i=1}^{n}(x_{i}-{\overline {x}})^{2}}{n}}$

# %% [markdown]
# The unbiased estimator would be:

# %% [markdown]
# $s=\sqrt {\frac {\sum _{i=1}^{n}(x_{i}-{\overline {x}})^{2}}{n-1}} = \sigma \cdot \sqrt{ \frac{n}{n-1}}$

# %% [markdown]
# The standard error is:
#
# $SE = \frac{\sigma}{n}$

# %% [markdown]
# Finally, we want 5-95th percentile. Assuming normal distribution, this amounts to multiplying the standard error by 1.645

# %%
std_2_95th = 1.645

# %%
import pandas as pd
num_mod_lab = 'Number of models (Thornhill 2020)'
thornhill = pd.read_csv('table2_thornhill2020.csv', index_col=0)
thornhill.index = thornhill.index.rename('Species')
thornhill

# %% [markdown]
# ![](thornhill.jpg)

# %%
sd_tot = table_sd['Total_sd']
df_err= pd.DataFrame(sd_tot.rename('std'))
df_err['SE'] = df_err

# %%
df_err['SE'] = df_err['std']/np.sqrt(thornhill[num_mod_lab])
df_err['95-50_SE'] = df_err['SE']*std_2_95th
df_err.loc['CO2','95-50_SE']= df_err.loc['CO2','std']
df_err

# %%
df_err['95-50'] = df_err['std']*std_2_95th
df_err.loc['CO2','95-50']= df_err.loc['CO2','std']
df_err

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
tab_plt = tab2.loc[::-1,var_dir].rename(rename_dic_cat, axis=1).rename(rename_dic_cols, axis=0)
tab_plt

# %%
tab_plt.sum()

# %%
tab_plt.sum(axis=1)

# %%

# %%
fn_sd = 'table_uncertainties_smb.csv'
fn_mean = 'table_mean_smb.csv'

# %%
df_err

# %%
df_err = df_err.rename(rename_dic_cols, axis=0)
df_err.to_csv(fn_sd)
tab_plt.to_csv(fn_mean)

# %%
width = 0.7
kwargs = {'linewidth':.1,'edgecolor':'k'}


# %%
import seaborn as sns

# %%
ybar = np.arange(len(tab_plt)+1)#, -1)
ybar

# %%
fig, ax = plt.subplots()#figsize=[10,10])


tab_plt.plot.barh(stacked=True, color=cols, linewidth=.0, edgecolor='k',ax=ax, width=width)
tot = table['Total'][::-1]
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

plt.savefig(fp, dpi=300)
plt.savefig(fp.with_suffix('.pdf'), dpi=300)
plt.show()


# %% [markdown]
# Will combine all uncertaintes

# %%
