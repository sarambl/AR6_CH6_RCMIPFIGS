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
import tqdm
from scmdata import ScmDataFrame, df_append

# %%
from ar6_ch6_rcmipfigs.utils.database_generation import check_all_variables_and_units_as_in_protocol, \
    check_all_scenarios_as_in_protocol, unify_units, save_into_database, mce_get_quantile, hector_get_quantile
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
model_of_interest = [
#    ".*acc2.*v2-0-1.*",
#    ".*rcmip_phase-1_cicero-scm.*v5-0-0.*",
#    ".*escimo.*v2-0-1.*",
#    ".*fair-1.5-default.*v1-0-1.csv",
#    ".*fair-1.5-ens.*v1-0-1.csv",
#    ".*rcmip_phase-1_gir.*",
#    ".*greb.*v2-0-0.*",
#    ".*hector.*v2-0-0.*",
#    ".*MAGICC7.1.0aX-rcmip-phase-1.*",
#    ".*rcmip_phase-1_magicc7.1.0.beta.*_v1-0-0.*",
#    ".*rcmip_phase-1_magicc7.1.0.beta.*_v1-0-0.*",
#    ".*MAGICC7.1.0aX.*",
#    ".*mce.*v2-0-1.*",
#    ".*oscar-v3-0*v1-0-1.*",
#    ".*oscar-v3-0.*v1-0-1.*"
#    ".*wasp.*v1-0-1.*",
]

# %%
magic_base = ".*rcmip_phase-1_magicc7.1.0.beta.*_v1-0-0.*"
mag_rf = get_res_fl(mod)
l_magicc_mods = []
for _a in mag_rf:
    print(_a.split('/')[-1])
    l_magicc_mods.append('.*'+_a.split('/')[-1])


# %%

# %%
for mod in model_of_interest + l_magicc_mods:
    print(mod)
    results_files = get_res_fl(mod)
    print(results_files)
    db = read_db(results_files)
    db = fix_ssp370_names(db)
    _assert(db)
    db = fix_hector_mce_mislab(db)
    db = fix_quant_FaIR(db)
    db = fix_quant_wasp(db)
    clean_db = unify_units_check_names(db, protocol_variables, protocol_scenarios)
    save_into_database(clean_db, OUTPUT_DATABASE_PATH, "rcmip-phase-1")

# %%
a=['/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-ipsl-cm6a-lr-r1i1p1f2_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-canesm5-r1i1p2f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-miroc6-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-giss-e2-2-g-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-bcc-esm1-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-cesm2-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-norcpm1-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-canesm5-r10i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-noresm2-lm-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-cnrm-cm6-1-hr-r1i1p1f2_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-giss-e2-1-h-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-sam0-unicon-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-cesm2-waccm-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-e3sm-1-0-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-ipsl-cm6a-lr-r10i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-fgoals-g3-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-ukesm1-0-ll-r1i1p1f2_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-mcm-ua-1-0-r1i1p1f2_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-bcc-csm2-mr-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-giss-e2-1-g-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-ipsl-cm6a-lr-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-ec-earth3-veg-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-miroc-es2l-r1i1p1f2_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-cnrm-esm2-1-r1i1p1f2_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-cnrm-cm6-1-r1i1p1f2_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-awi-cm-1-1-mr-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-mpi-esm1-2-hr-r1i1p1f1_v1-0-0.csv', '/home/sarambl/PHD/IPCC/public/AR6_CH6_RCMIPFIGS/ar6_ch6_rcmipfigs/data_in/data/results/phase-1/magicc7/rcmip_phase-1_magicc7.1.0.beta-canesm5-r1i1p1f1_v1-0-0.csv']

# %%
l_magicc_mods = []
for _a in a:
    print(_a.split('/')[-1])
    l_magicc_mods.append('.*'+_a.split('/')[-1])

