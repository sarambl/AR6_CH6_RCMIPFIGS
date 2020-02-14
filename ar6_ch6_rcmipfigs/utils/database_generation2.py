import os
import re
from pathlib import Path

import tqdm
from scmdata import ScmDataFrame, df_append

from ar6_ch6_rcmipfigs.utils.database_generation import mce_get_quantile, hector_get_quantile, unify_units, \
    check_all_scenarios_as_in_protocol, check_all_variables_and_units_as_in_protocol

from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR

def get_res_fl(mod, resultpath=None):
    if resultpath is None:
        resultpath= os.path.join(INPUT_DATA_DIR, "data", "results", "phase-1")
    _results_files = list(Path(resultpath).rglob("*.csv")) + list(
        Path(resultpath).rglob("*.xlsx"))
    results_files = [str(p)
                     for p in _results_files
                     if bool(re.match(mod, str(p))) and "$" not in str(p)]
    return results_files


def read_db(results_files):
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
    return db


def fix_ssp370_names(db):
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
    return db


def _assert(db):
    assert "ssp370-lowNTCF" not in db["scenario"].unique().tolist()
    assert "esm-ssp370-lowNTCF" not in db["scenario"].unique().tolist()
    assert "esm-ssp370-lowNTCF-allGHG" not in db["scenario"].unique().tolist()


def fix_hector_mce_mislab(db):
    mce_prob_data = db.filter(climatemodel="MCE*PROB*")
    mce_prob_data["climatemodel"].unique()
    if not mce_prob_data.timeseries().empty:
        mce_prob_data = mce_prob_data.timeseries().reset_index()

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

    hector_prob_data = db.filter(climatemodel="hector*HISTCALIB*")
    if not hector_prob_data.timeseries().empty:
        hector_prob_data = hector_prob_data.timeseries().reset_index()

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
    return db


def fix_quant_FaIR(db):
    fair_prob_data = db.filter(climatemodel="*FaIR*")

    if not fair_prob_data.timeseries().empty:
        fair_prob_data = fair_prob_data.timeseries().reset_index()

        fair_prob_data["variable"] = fair_prob_data["variable"].apply(
            lambda x: x.replace("|00th", "|0th").replace("|05th", "|5th")
        )

        db = db.filter(climatemodel="*FaIR*", keep=False).append(
            ScmDataFrame(fair_prob_data)
        )
    return db


def fix_quant_wasp(db):
    wasp_prob_data = db.filter(climatemodel="*WASP*")
    if not wasp_prob_data.timeseries().empty:
        wasp_prob_data = wasp_prob_data.timeseries().reset_index()

        wasp_prob_data["variable"] = wasp_prob_data["variable"].apply(
            lambda x: x.replace("|00th", "|0th").replace("|05th", "|5th")
        )

        db = db.filter(climatemodel="*WASP*", keep=False).append(
            ScmDataFrame(wasp_prob_data)
        )
    return db


def unify_units_check_names(db, protocol_variables,protocol_scenarios):
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
        except AssertionError as e:
            print(e)
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
    return clean_db