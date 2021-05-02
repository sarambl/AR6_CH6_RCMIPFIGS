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

# %%
runpy.run_path('01_preprocess.py')
clear_output()

# %% [markdown]
# ## Create $\Delta$GSAT:

# %%
print('1')
runpy.run_path('02-01_compute_delta_T_recommandation.py')
print('2***********')

runpy.run_path('02-01-02_compute_delta_T_recommandation-HFCs.py')

runpy.run_path('02-02_uncertainty.py')

runpy.run_path('02-03_uncertainty_sum_aerosols.py')
print('done')


# %% [markdown]
#
# ## Create plots etc:

# %%
runpy.run_path('03-02_delta_T_plot_bar_stacked_recommendation.py')
print('done')
# %% [markdown]
#
