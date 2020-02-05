from pprint import pprint

import pyam
import tqdm
from scmdata import ScmDataFrame

from ar6_ch6_rcmipfigs.utils.misc_func import logger


def subtract(base, other, axis, new_name):
    b_data = base.timeseries()
    o_data = other.timeseries()

    idx = b_data.index.names
    idx_tmp = list(set(idx) - {axis})

    b_data = b_data.reset_index().set_index(idx_tmp).drop(axis, axis="columns")
    o_data = o_data.reset_index().set_index(idx_tmp).drop(axis, axis="columns")

    res = (b_data - o_data).reset_index()
    res[axis] = new_name

    return pyam.IamDataFrame(res)


def convert_scmdf_to_pyamdf_year_only(iscmdf):
    out = iscmdf.timeseries()
    out.columns = out.columns.map(lambda x: x.year)

    return pyam.IamDataFrame(out)


def check_all_variables_and_units_as_in_protocol(df_to_check, protocol_variables):
    checker_df = df_to_check.filter(variable="*Other|*", keep=False)[
        ["variable", "unit"]
    ]
    checker_df.columns.name = None
    checker_df["unit"] = checker_df["unit"].apply(
        lambda x: x.replace("dimensionless", "Dimensionless")
        if isinstance(x, str)
        else x
    )

    def strip_quantile(inv):
        if inv.endswith("quantile"):
            return "|".join(inv.split("|")[:-1])

        return inv

    checker_df["variable"] = checker_df["variable"].apply(strip_quantile)
    merged_df = checker_df.merge(
        protocol_variables[["variable", "unit"]], how="inner", on=["variable", "unit"]
    ).drop_duplicates()
    checker_df = checker_df.drop_duplicates()
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
    #TODO compare to other version in database_generation.py
    out_df = in_df.copy()
    for variable in tqdm.tqdm_notebook(out_df["variable"].unique()):
        if variable.startswith("Radiative Forcing|Anthropogenic|Other"):
            target_unit = protocol_variables[
                protocol_variables["variable"]
                == "Radiative Forcing|Anthropogenic|Other"
            ]["unit"].iloc[0]
        elif variable.endswith("quantile"):
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