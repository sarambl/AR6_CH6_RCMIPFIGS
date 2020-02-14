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
# ## OBS:
# This notebook is only slightly edited from Zebedee Nicholls notebook, see [here](https://gitlab.com/rcmip/rcmip/-/blob/master/notebooks/results/phase-1/database-generation.ipynb)

# %%
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR

__depends__ = []
__dest__ = [INPUT_DATA_DIR+
    "/data/database-results/phase-1/timestamp.txt",
    INPUT_DATA_DIR+"/data/database-observations/timestamp.txt",
]

# %%
# %load_ext nb_black

# %% [markdown]
# # Database generation
#
#
# In this notebook we process the data into a database we can later query to make plots/do analysis etc.

# %% [markdown]
# ## Imports

# %%
import logging
import os.path
import re
from pathlib import Path
from distutils.util import strtobool

import pandas as pd
from scmdata import ScmDataFrame

# %%
from ar6_ch6_rcmipfigs.utils.database_generation import save_into_database
from ar6_ch6_rcmipfigs.utils.database_generation2 import get_res_fl, read_db, fix_ssp370_names, _assert, \
    fix_hector_mce_mislab, fix_quant_FaIR, fix_quant_wasp, unify_units_check_names

TEST_RUN = strtobool(os.getenv("CI", "False")) or False
TEST_RUN

# %%
logger = logging.getLogger()

# %% [markdown]
# ## Constants

# %%
from ar6_ch6_rcmipfigs.constants import  INPUT_DATA_DIR
OUTPUT_DATABASE_PATH = os.path.join(INPUT_DATA_DIR, "database-results", "phase-1/")

OBS_DATABASE_PATH = os.path.join(INPUT_DATA_DIR, "database-observations/")

# %%
from ar6_ch6_rcmipfigs.utils.misc_func import make_folders

if not os.path.isdir(OUTPUT_DATABASE_PATH):
    make_folders(OUTPUT_DATABASE_PATH)

if not os.path.isdir(OBS_DATABASE_PATH):
    make_folders(OBS_DATABASE_PATH)


# %% [markdown]
# ## Protocol

# %%
SCENARIO_PROTOCOL = os.path.join(INPUT_DATA_DIR, "data", "protocol", "rcmip-emissions-annual-means.csv"
)

# %%
protocol_db = ScmDataFrame(SCENARIO_PROTOCOL)
protocol_db.head()

# %%
protocol_db["scenario"].unique()

# %%
DATA_PROTOCOL = os.path.join(INPUT_DATA_DIR,
    "data",
    "submission-template",
    "rcmip-data-submission-template.xlsx",
)

# %%
protocol_variables = pd.read_excel(DATA_PROTOCOL, sheet_name="variable_definitions")
protocol_variables.columns = protocol_variables.columns.str.lower()
protocol_variables.head()

# %%
protocol_scenarios = pd.read_excel(
    DATA_PROTOCOL, sheet_name="scenario_info", skip_rows=2
)
protocol_scenarios.columns = protocol_scenarios.columns.str.lower()
protocol_scenarios.head()

# %% [markdown]
# ## Model output

# %%
RESULTS_PATH = os.path.join(INPUT_DATA_DIR, "data", "results", "phase-1")

# %%


# %%


# %%
model_of_interest = [
#    ".*acc2.*v2-0-1.*",
    ".*rcmip_phase-1_cicero-scm.*v5-0-0.*",
#    ".*escimo.*v2-0-1.*",
#    ".*fair-1.5-default.*v1-0-1.csv",
#    ".*rcmip_phase-1_gir.*",
#    ".*greb.*v2-0-0.*",
#    ".*hector.*v2-0-0.*",
#    ".*MAGICC7.1.0aX-rcmip-phase-1.*",
#    ".*rcmip_phase-1_magicc7.1.0.beta_v1-0-0.*",
#    ".*MAGICC7.1.0aX.*",
#    ".*mce.*v2-0-1.*",
#    ".*oscar-v3-0*v1-0-1.*",
#    ".*oscar-v3-0.*v1-0-1.*"
#    ".*wasp.*v1-0-1.*",
]

# %%

results_files =[]
for mod in model_of_interest:
    l = get_res_fl(mod, resultpath=RESULTS_PATH)
    results_files = results_files+l
# %%
db = read_db(results_files)
db.head()

# %%
db.filter(climatemodel="*Cicero*").head()

# %%
db["climatemodel"].unique()

# %% [markdown]
# ### Minor quick fixes

# %% [markdown]
# We relabel all the ssp370-lowNTCF data to remove ambiguity.

# %%
db = fix_ssp370_names(db)
# %%
_assert(db)
# %% [markdown]
# The Hector and MCE data is mislabelled so we do a quick fix here. I also have changed my mind about how to format the quantiles so tweak the FaIR and WASP data too.

# %%


db = fix_hector_mce_mislab(db)
db.filter(climatemodel="MCE*PROB").head(10)
# %%
db.filter(climatemodel="hector*HISTCALIB").head(10)

# %%
db= fix_quant_FaIR(db)
db.filter(climatemodel="*FaIR*").head(10)

# %%
db= fix_quant_wasp(db)
db.filter(climatemodel="*WASP*").head(10)


# %% [markdown]
# ## Unify units and check names
#
# Here we loop over the submissions and unify their units as well as checking their naming matches what we expect.

# %%
clean_db = unify_units_check_names(db, protocol_variables, protocol_scenarios)
# %%

clean_db.head()

# %% [markdown]
# Notes whilst doing this:
#
# - I wasn't clear that the variable hierarchy needs to be obeyed, hence doing internal consistency checks isn't going to work
#
# For phase 2:
#
# - checking internal consistency super slow, worth looping over top level variables when doing this to speed up filtering
# - need to decide what a sensible tolerance is
# - might have to go back to model notes to work out why there are inconsistencies
# - will have to implement a custom hack to deal with the double counting in the direct aerosol forcing hierarchy

# %% [markdown]
# ## Creating a database

# %%
save_into_database(clean_db, OUTPUT_DATABASE_PATH, "rcmip-phase-1")

# %%

for mod in model_of_interest:
    results_files = get_res_fl(mod)
    db = read_db(results_files)
    db = fix_ssp370_names(db)
    _assert(db)
    db = fix_hector_mce_mislab(db)
    db = fix_quant_FaIR(db)
    db = fix_quant_wasp(db)
    clean_db = unify_units_check_names(db, protocol_variables, protocol_scenarios)
    save_into_database(clean_db, OUTPUT_DATABASE_PATH, "rcmip-phase-1")