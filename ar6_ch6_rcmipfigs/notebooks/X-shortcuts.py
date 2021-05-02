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
from IPython.display import clear_output

# %%
import runpy

# %% [markdown]
# ## Preprocess:
#
# ## Figures ERF emission based attribution (from Bill Collins script)

# %%
runpy.run_path('ERF_hist_attribution/make_historical_attribution.py')
print('done')

# %% [markdown]
# ## Figures ERF/GSAT emission based attribution:

# %%
runpy.run_path('GSAT_change_hist_attribution/01_Emission_based_ERFs.py')
runpy.run_path('GSAT_change_hist_attribution/02_historical_deltaGSAT.py')
runpy.run_path('GSAT_change_hist_attribution/03_plot.py')
print('done')
