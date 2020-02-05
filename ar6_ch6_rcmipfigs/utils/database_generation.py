import os
import re
from datetime import datetime
from pprint import pprint
import logging
import pyam
import tqdm
from scmdata import ScmDataFrame
"""
All code is based on or directly copied from Zebedee Nicholls (zebedee.nicholls@climate-energy-college.org)
 code https://gitlab.com/rcmip/rcmip
"""



logger = logging.getLogger()
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


def check_all_scenarios_as_in_protocol(df_to_check, protocol_scenarios):
    checker_df = df_to_check["scenario"].to_frame()
    merged_df = checker_df.merge(protocol_scenarios[["scenario"]])
    assert len(merged_df) == len(checker_df), set(checker_df["scenario"]) - set(
        merged_df["scenario"]
    )


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


def convert_scmdf_to_pyamdf_year_only(iscmdf):
    out = iscmdf.timeseries()
    out.columns = out.columns.map(lambda x: x.year)

    return pyam.IamDataFrame(out)


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


def mce_get_quantile(inp):
    if inp.endswith("33rd"):
        return "33"

    if inp.endswith("67th"):
        return "67"

    if inp.endswith("17th"):
        return "17"

    if inp.endswith("83rd"):
        return "83"

    if inp.endswith("50th"):
        return "50"

    raise NotImplementedError(inp)


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