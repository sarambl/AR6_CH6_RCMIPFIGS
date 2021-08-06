import re
from pprint import pprint

import pyam

#from scmdata import ScmDataFrame

"""
All code is based on or directly copied from Zebedee Nicholls (zebedee.nicholls@climate-energy-college.org)
 code https://gitlab.com/rcmip/rcmip
"""



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
