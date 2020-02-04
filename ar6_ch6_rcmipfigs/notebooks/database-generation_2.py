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
__depends__ = []
__dest__ = [
    "../../../data/database-results/phase-1/timestamp.txt",
    "../../../data/database-observations/timestamp.txt",
]

# %%
# %load_ext nb_black

# %% [markdown]
# # Database generation
#
# In this notebook we process the data into a database we can later query to make plots/do analysis etc.

# %% [markdown]
# ## Imports

# %%
import logging
import os
import os.path
import re
from datetime import datetime
from pathlib import Path
from pprint import pprint
from time import sleep
from distutils.util import strtobool

import pandas as pd
import pyam
import tqdm
from scmdata import ScmDataFrame, df_append

# %%
TEST_RUN = strtobool(os.getenv("CI", "False")) or False
TEST_RUN

# %%
logger = logging.getLogger()

# %% [markdown]
# ## Constants

# %%
OUTPUT_DATABASE_PATH = os.path.join(
    "..", "..", "..", "data", "database-results", "phase-1"
)

OBS_DATABASE_PATH = os.path.join("..", "..", "..", "data", "database-observations")

# %%
if not os.path.isdir(OUTPUT_DATABASE_PATH):
    os.mkdir(OUTPUT_DATABASE_PATH)

if not os.path.isdir(OBS_DATABASE_PATH):
    os.mkdir(OBS_DATABASE_PATH)


# %% [markdown]
# ## Miscellaneous functions
#
# TODO: put these into some sort of `utils` file

# %%
def strip_quantile(inv):
    if inv.endswith("mean"):
        return "|".join(inv.split("|")[:-1])

    if inv.endswith("stddev"):
        return "|".join(inv.split("|")[:-1])

    if "quantile" in inv:
        if re.match(".*\|([1-9]\d?|100|0|[1-9]\d*\.\d)th quantile$", inv) is None:
            print("Bad formatting: {}".format(inv))

        return "|".join(inv.split("|")[:-1])

    return inv


def check_all_variables_and_units_as_in_protocol(df_to_check, protocol_variables):
    checker_df = df_to_check.filter(variable="*Other*", keep=False)[
        ["variable", "unit"]
    ]
    checker_df["unit"] = checker_df["unit"].apply(
        lambda x: x.replace("dimensionless", "Dimensionless")
        if isinstance(x, str)
        else x
    )

    def strip_quantile(inv):
        if any([inv.endswith(suf) for suf in ["quantile", "mean", "stddev"]]):
            return "|".join(inv.split("|")[:-1])

        return inv

    checker_df["variable"] = checker_df["variable"].apply(strip_quantile)
    merged_df = checker_df.merge(protocol_variables[["variable", "unit"]])
    try:
        assert len(merged_df) == len(checker_df)
    except AssertionError:
        pprint(set(checker_df["variable"]) - set(protocol_variables["variable"]))
        pprint(set(checker_df["unit"]) - set(protocol_variables["unit"]))
        raise


# %%
def check_all_scenarios_as_in_protocol(df_to_check, protocol_scenarios):
    checker_df = df_to_check["scenario"].to_frame()
    merged_df = checker_df.merge(protocol_scenarios[["scenario"]])
    assert len(merged_df) == len(checker_df), set(checker_df["scenario"]) - set(
        merged_df["scenario"]
    )


# %%
def unify_units(in_df, protocol_variables, exc_info=False):
    out_df = in_df.copy()
    for variable in tqdm.tqdm_notebook(out_df["variable"].unique()):
        if variable.startswith("Radiative Forcing|Anthropogenic|Albedo Change"):
            target_unit = protocol_variables[
                protocol_variables["variable"]
                == "Radiative Forcing|Anthropogenic|Albedo Change"
            ]["unit"].iloc[0]

        elif variable.startswith(
            "Effective Radiative Forcing|Anthropogenic|Albedo Change"
        ):
            target_unit = protocol_variables[
                protocol_variables["variable"]
                == "Effective Radiative Forcing|Anthropogenic|Albedo Change"
            ]["unit"].iloc[0]

        elif variable.startswith("Carbon Pool"):
            target_unit = protocol_variables[
                protocol_variables["variable"] == "Carbon Pool|Atmosphere"
            ]["unit"].iloc[0]

        elif "Other" in variable:
            target_unit = protocol_variables[
                protocol_variables["variable"]
                == "{}".format(variable.split("|Other")[0])
            ]["unit"].iloc[0]

        elif any([variable.endswith(suf) for suf in ["quantile", "mean", "stddev"]]):
            try:
                target_unit = protocol_variables[
                    protocol_variables["variable"] == "|".join(variable.split("|")[:-1])
                ]["unit"].iloc[0]
            except:
                logger.exception(
                    f"Failed to find unit for {variable}", exc_info=exc_info
                )
                continue
        else:
            try:
                target_unit = protocol_variables[
                    protocol_variables["variable"] == variable
                ]["unit"].iloc[0]
            except:
                logger.exception(
                    f"Failed to find unit for {variable}", exc_info=exc_info
                )
                continue

        try:
            if "CH4" in target_unit:
                out_df = out_df.convert_unit(
                    target_unit, variable=variable, context="CH4_conversions"
                )
                continue

            if "NOx" in target_unit:
                out_df = out_df.convert_unit(
                    target_unit, variable=variable, context="NOx_conversions"
                )
                continue

            if target_unit == "Dimensionless":
                target_unit = "dimensionless"

            out_df = out_df.convert_unit(target_unit, variable=variable)
        except:
            current_unit = out_df.filter(variable=variable)["unit"].unique()
            logger.exception(
                f"Failed for {variable} with target unit: {target_unit} and current_unit: {current_unit}",
                exc_info=exc_info,
            )

    out_df = out_df.timeseries().reset_index()
    out_df["unit_context"] = out_df["unit_context"].fillna("not_required")
    return ScmDataFrame(out_df)


# %%
def aggregate_variable(db_in, v_to_agg):
    v_to_agg_df = (
        db_in.filter(variable=v_to_agg, keep=False)
        .filter(
            variable="{}|*".format(v_to_agg),
            level=0,  # make sure we don't pick up e.g. HFC23|50th Percentile by accident
        )
        .timeseries()
    )
    group_idx = list(set(v_to_agg_df.index.names) - {"variable"})
    v_to_agg_df = v_to_agg_df.groupby(group_idx).sum().reset_index()
    v_to_agg_df["variable"] = v_to_agg

    db_out = db_in.append(v_to_agg_df)

    return db_out


# %%
def prep_str_for_filename(ins):
    return (
        ins.replace("_", "-")
        .replace("|", "-")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def get_filename(scmdf, leader):
    climatemodel = prep_str_for_filename(
        scmdf.get_unique_meta("climatemodel", no_duplicates=True)
    )
    variable = prep_str_for_filename(
        scmdf.get_unique_meta("variable", no_duplicates=True)
    )
    region = prep_str_for_filename(scmdf.get_unique_meta("region", no_duplicates=True))

    return "{}_{}_{}_{}.csv".format(leader, climatemodel, region, variable)


# %%
def convert_scmdf_to_pyamdf_year_only(iscmdf):
    out = iscmdf.timeseries()
    out.columns = out.columns.map(lambda x: x.year)

    return pyam.IamDataFrame(out)


# %%
def save_into_database(db, db_path, filename_leader):
    for cm in tqdm.tqdm_notebook(
        db["climatemodel"].unique(), leave=False, desc="Climate models"
    ):
        db_cm = db.filter(climatemodel=cm)
        for r in tqdm.tqdm_notebook(
            db_cm["region"].unique(), leave=False, desc="Regions"
        ):
            db_cm_r = db_cm.filter(region=r)
            for v in tqdm.tqdm_notebook(
                db_cm_r["variable"].unique(), leave=False, desc="Variables"
            ):
                db_cm_r_v = ScmDataFrame(db_cm_r.filter(variable=v))
                filename = get_filename(db_cm_r_v, leader=filename_leader)
                outfile = os.path.join(db_path, filename)

                convert_scmdf_to_pyamdf_year_only(db_cm_r_v).to_csv(outfile)
                logger.debug("saved file to {}".format(outfile))

    with open(os.path.join(db_path, "timestamp.txt"), "w") as fh:
        fh.write("database written at: ")
        fh.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fh.write("\n")


# %% [markdown]
# ## Protocol

# %%
SCENARIO_PROTOCOL = os.path.join(
    "..", "..", "..", "data", "protocol", "rcmip-emissions-annual-means.csv"
)

# %%
protocol_db = ScmDataFrame(SCENARIO_PROTOCOL)
protocol_db.head()

# %%
protocol_db["scenario"].unique()

# %%
DATA_PROTOCOL = os.path.join(
    "..",
    "..",
    "..",
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
# ## Observations

# %% [markdown]
# ### tas observations
#
# These come from Chris Smith (personal email). TODO: get details from Chris about what these are.

# %%
TAS_OBS_PATH = os.path.join(
    "..", "..", "..", "data", "priestley-centre", "observations", "tas_obs.txt"
)

# %%
tas_obs_df = pd.read_csv(TAS_OBS_PATH, header=None, delim_whitespace=True)
tas_obs_df.columns = ["year", "value"]
tas_obs_df["model"] = "unspecified"
tas_obs_df["climatemodel"] = "Observations (Priestley Centre)"
tas_obs_df["scenario"] = "historical"
tas_obs_df["variable"] = "Surface Air Temperature Change"
tas_obs_df["unit"] = "K"
tas_obs_df["region"] = "World"
tas_obs_df = pyam.IamDataFrame(tas_obs_df)
tas_obs_df.head()

# %%
db_obs = pyam.concat([tas_obs_df])
db_obs.head()

# %%
save_into_database(db_obs, OBS_DATABASE_PATH, "rcmip-observations")

# %% [markdown]
# ## Model output

# %%
RESULTS_PATH = os.path.join("..", "..", "..", "data", "results", "phase-1")

# %%
results_files = list(Path(RESULTS_PATH).rglob("*.csv")) + list(
    Path(RESULTS_PATH).rglob("*.xlsx")
)
print(len(results_files))
sorted(results_files)

# %%
model_of_interest = [
#    ".*acc2.*v2-0-1.*",
    ".*rcmip_phase-1_cicero-scm.*v5-0-0.*",
#    ".*escimo.*v2-0-1.*",
    ".*fair-1.5-default.*",
#    ".*rcmip_phase-1_gir.*",
#    ".*greb.*v2-0-0.*",
#    ".*hector.*v2-0-0.*",
#    ".*MAGICC7.1.0aX-rcmip-phase-1.*",
    ".*rcmip_phase-1_magicc7.1.0.beta.*"
#    ".*MAGICC7.1.0aX.*",
#    ".*mce.*v2-0-1.*",
    ".*oscar.*",
#    ".*wasp.*v1-0-1.*",
]
if TEST_RUN:
    model_of_interest = [
        ".*escimo-phase-1-v2-0-1.*",
        ".*greb.*",
        ".*rcmip_phase-1_cicero-scm.*v5-0-0.*",
    ]

results_files = [
    str(p)
    for p in results_files
    if any([bool(re.match(m, str(p))) for m in model_of_interest]) and "$" not in str(p)
]
print(len(results_files))
sorted(results_files)

# %%
db = []
for rf in tqdm.tqdm_notebook(results_files):
    if rf.endswith(".csv"):
        loaded = ScmDataFrame(rf)
    else:
        loaded = ScmDataFrame(rf, sheet_name="your_data")
    db.append(loaded)

db = df_append(db).timeseries().reset_index()
db["unit"] = db["unit"].apply(
    lambda x: x.replace("Dimensionless", "dimensionless") if isinstance(x, str) else x
)
db = ScmDataFrame(db)
db.head()

# %%
db["climatemodel"].unique()

# %% [markdown]
# ### Minor quick fixes

# %% [markdown]
# We relabel all the ssp370-lowNTCF data to remove ambiguity.

# %%
db = db.timeseries().reset_index()
db["scenario"] = db["scenario"].apply(
    lambda x: "ssp370-lowNTCF-gidden" if x == "ssp370-lowNTCF" else x
)
db["scenario"] = db["scenario"].apply(
    lambda x: "esm-ssp370-lowNTCF-gidden" if x == "esm-ssp370-lowNTCF" else x
)
db["scenario"] = db["scenario"].apply(
    lambda x: "esm-ssp370-lowNTCF-gidden-allGHG"
    if x == "esm-ssp370-lowNTCF-allGHG"
    else x
)
db = ScmDataFrame(db)

# %%
assert "ssp370-lowNTCF" not in db["scenario"].unique().tolist()
assert "esm-ssp370-lowNTCF" not in db["scenario"].unique().tolist()
assert "esm-ssp370-lowNTCF-allGHG" not in db["scenario"].unique().tolist()

# %% [markdown]
# The Hector and MCE data is mislabelled so we do a quick fix here. I also have changed my mind about how to format the quantiles so tweak the FaIR and WASP data too. TODO: email modelling groups so they can fix it for phase 2.

# %%
mce_prob_data = db.filter(climatemodel="MCE*PROB*")
mce_prob_data["climatemodel"].unique()
if not mce_prob_data.timeseries().empty:
    mce_prob_data = mce_prob_data.timeseries().reset_index()

    def mce_get_quantile(inp):
        if inp.endswith("33rd"):
            return "33"

        if inp.endswith("67th"):
            return "67"

        raise NotImplementedError

    mce_prob_data["variable"] = (
        mce_prob_data["variable"]
        + "|"
        + mce_prob_data["climatemodel"].apply(mce_get_quantile)
        + "th quantile"
    )

    mce_prob_data["climatemodel"] = mce_prob_data["climatemodel"].apply(
        lambda x: "-".join(x.split("-")[:-1])
    )

    db = db.filter(climatemodel="MCE*PROB*", keep=False).append(mce_prob_data)

db.filter(climatemodel="MCE*PROB").head(10)

# %%
hector_prob_data = db.filter(climatemodel="hector*HISTCALIB*")
if not hector_prob_data.timeseries().empty:
    hector_prob_data = hector_prob_data.timeseries().reset_index()

    def hector_get_quantile(inp):
        if inp.endswith("SD"):
            return "stddev"
        if inp.endswith("Mean"):
            return "mean"

        tmp = inp.split("q")[1]
        if len(tmp) == 3:
            tmp = "{}.{}".format(tmp[:2], tmp[2])
        if tmp.startswith("0"):
            tmp = tmp[1:]
        return tmp + "th quantile"

    hector_prob_data["variable"] = (
        hector_prob_data["variable"]
        + "|"
        + hector_prob_data["climatemodel"].apply(hector_get_quantile)
    )

    hector_prob_data["climatemodel"] = hector_prob_data["climatemodel"].apply(
        lambda x: x.split("-")[0]
    )

    db = db.filter(climatemodel="hector*HISTCALIB*", keep=False).append(
        hector_prob_data
    )

db.filter(climatemodel="hector*HISTCALIB").head(10)

# %%
fair_prob_data = db.filter(climatemodel="*FaIR*")
if not fair_prob_data.timeseries().empty:
    fair_prob_data = fair_prob_data.timeseries().reset_index()

    fair_prob_data["variable"] = fair_prob_data["variable"].apply(
        lambda x: x.replace("|00th", "|0th").replace("|05th", "|5th")
    )

    db = db.filter(climatemodel="*FaIR*", keep=False).append(
        ScmDataFrame(fair_prob_data)
    )

db.filter(climatemodel="*FaIR*").head(10)

# %%
wasp_prob_data = db.filter(climatemodel="*WASP*")
if not wasp_prob_data.timeseries().empty:
    wasp_prob_data = wasp_prob_data.timeseries().reset_index()

    wasp_prob_data["variable"] = wasp_prob_data["variable"].apply(
        lambda x: x.replace("|00th", "|0th").replace("|05th", "|5th")
    )

    db = db.filter(climatemodel="*WASP*", keep=False).append(
        ScmDataFrame(wasp_prob_data)
    )

db.filter(climatemodel="*WASP*").head(10)

# %% [markdown]
# ## Unify units and check names
#
# Here we loop over the submissions and unify their units as well as checking their naming matches what we expect.

# %%
base_df = db.timeseries()
any_failures = False

clean_db = []
for climatemodel, cdf in tqdm.tqdm_notebook(
    base_df.groupby("climatemodel"), desc="Climate model"
):
    print(climatemodel)
    print("-" * len(climatemodel))

    any_failures_climatemodel = False

    cdf = ScmDataFrame(cdf)
    cdf_converted_units = unify_units(cdf, protocol_variables)
    try:
        check_all_scenarios_as_in_protocol(cdf_converted_units, protocol_scenarios)
        check_all_variables_and_units_as_in_protocol(
            cdf_converted_units, protocol_variables
        )
    except AssertionError:
        any_failures_climatemodel = True
    #     # currently not possible as groups weren't told to obey variable hierarchy,
    #     # add this in phase 2
    #     for v_top in cdf_converted_units.filter(level=0)["variable"].unique():
    #         print(v_top)
    #         cdf_pyam = cdf_converted_units.filter(variable="{}*".format(v_top)).timeseries()
    #         cdf_pyam.columns = cdf_pyam.columns.map(lambda x: x.year)

    #         cdf_consistency_checker = pyam.IamDataFrame(cdf_pyam)
    #         if cdf_consistency_checker.check_internal_consistency() is not None:
    #             print("Failed for {}".format(v_top))
    #             any_failures_climatemodel = True
    #             failing_set = cdf_consistency_checker.copy()

    print()
    if not any_failures_climatemodel:
        clean_db.append(cdf_converted_units)
        print("All clear for {}".format(climatemodel))
    else:
        print("Failed {}".format(climatemodel))
        print("X" * len("Failed"))
        any_failures = True

    print()
    print()

if any_failures:
    raise AssertionError("database isn't ready yet")
else:
    clean_db = df_append(clean_db)
    clean_db.head()

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
