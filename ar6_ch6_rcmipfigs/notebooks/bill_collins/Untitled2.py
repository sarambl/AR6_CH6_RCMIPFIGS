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
import attribution_1750_2019_v2_smb

# %%
table, table_sd = attribution_1750_2019_v2_smb.main()

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_chem_col

# %%
varn = ['co2','WMGHG','o3','ch4','H2O_strat','ari','aci']
var_dir = ['CO2','GHG','CH4_lifetime','O3','Strat_H2O','Aerosol','Cloud']

# %%
cols = [get_chem_col(var) for var in varn]

# %%

# %%
tab2[var_dir]

# %%

# %%

o3_sum = table['O3']+table['O3_prime']
tab2 = table.copy(deep=True).drop(['O3','O3_prime','Total'], axis=1)
tab2['O3'] = o3_sum
tab2.loc[::-1,var_dir].plot.barh(stacked=True, color=cols, linewidth=.3, edgecolor='k',
                                
                                )


# %% [markdown]
# ## Uncertainty:

# %%
table_sd
