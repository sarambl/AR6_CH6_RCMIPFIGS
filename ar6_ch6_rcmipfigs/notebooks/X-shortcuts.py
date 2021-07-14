# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from IPython.display import clear_output

# %%
import runpy

# %% [markdown]
# ## Preprocess:

# %%

# %% [markdown]
# ## Figures ERF emission based attribution (from Bill Collins script)

# %%
runpy.run_path('GSAT_change_hist_attribution/01_make_historical_attribution.py')
print('done')

# %% [markdown]
# ## Figures ERF/GSAT emission based attribution:

# %% tags=[]
runpy.run_path('GSAT_change_hist_attribution/02_Emission_based_ERFs.py')
runpy.run_path('GSAT_change_hist_attribution/03_historical_deltaGSAT.py')
runpy.run_path('GSAT_change_hist_attribution/04_01_plot-period.py')
runpy.run_path('GSAT_change_hist_attribution/04_02_plot.py')
print('done')

# %%
