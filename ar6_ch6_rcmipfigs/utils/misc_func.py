import logging
from pprint import pprint
import pandas as pd
import pyam
import seaborn as sns
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scmdata import df_append, ScmDataFrame
from ar6_ch6_rcmipfigs.constants import BASE_DIR
import matplotlib.pyplot as plt

climatemodel='climatemodel'
logger = logging.getLogger()

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



def aggregate_variable(db_in, v_to_agg, cmodel, remove_quantiles=True):
    # %%
    #remove_quantiles=True
    #db_in = db_converted_units.copy()
    #v_to_agg = erf_aerosols
    #cmodel = 'MAGICC7.1.0.beta-rcmip-phase-1'
    if not remove_quantiles:
        raise NotImplementedError("quantile handling wrong")

    _db = db_in.filter(variable='*quantile', keep=False).filter(climatemodel=cmodel)
    for cm in _db[climatemodel].unique():
        if len(_db.filter(variable=v_to_agg, climatemodel=cm)['scenario'].unique())>0:
            print('Model %s already has variable, returns unchanged'%cmodel)
            return db_in

    #print(_db.head())
    # remove models that already have the variable:
    #_db_check = _db.filter(variable=v_to_agg)
    #print(_db.head())


    v_to_agg_df = (
        _db.filter(variable=v_to_agg, keep=False) # remove v_to_agg
            .filter(
            variable="{}|*".format(v_to_agg), # pick out subgroups.
            level=0,  # make sure we don't pick up e.g. HFC23|50th Percentile by accident
        )
            .timeseries()
    )
    if len(v_to_agg_df)==0:
        print('No variables to aggregate found in model %s. Returns unchanged'%cmodel)
        return db_in
    # %%
    # %%
    #print(v_to_agg_df)
    group_idx = list(set(v_to_agg_df.index.names) - {"variable"})
    # %%
    groups = v_to_agg_df.groupby(group_idx).groups
    for g in groups.keys():
        print('**Aggragating for model %s:**'%cmodel)
        print('Group:')
        print(g)
        print('Aggregates:')
        print(list(groups[g]))

    v_to_agg_df = v_to_agg_df.groupby(group_idx).sum().reset_index()
    v_to_agg_df["variable"] = v_to_agg
    #print(v_to_agg_df)
    # %%
    db_out = db_in.append(v_to_agg_df)
    # To make sure we use the pre-calculated version IF IT EXISTS,
    # we replace these values:

    return db_out


def fix_BC_name(db_in,
                from_v = 'Effective Radiative Forcing|Anthropogenic|Albedo Change|Other|Deposition of Black Carbon on Snow',
                to_v = 'Effective Radiative Forcing|Anthropogenic|Other|BC on Snow',
                model= "*OSCAR*"):
    #print(_db.head())
    # remove models that already have the variable:
    #_db_check = _db.filter(variable=v_to_agg)
    #print(_db.head())
    db = db_in.timeseries().reset_index()
    db["variable"] = db["variable"].apply(lambda x: to_v if x == from_v else x)
    db = ScmDataFrame(db)
    return db

    v_ren = (
        db_in.filter(variable=from_v, climatemodel=model)#, keep=False) # remove v_to_agg
        #.filter(
        #    variable="{}|*".format(v_to_agg), # pick out subgroups.
        #    level=0,  # make sure we don't pick up e.g. HFC23|50th Percentile by accident
        #)
        .timeseries()
    )
    #print(v_to_agg_df)
    #group_idx = list(set(v_ren.index.names) - {"variable"})
    #v_ren = v_ren.groupby(group_idx).sum().reset_index()
    v_ren = v_ren.reset_index()
    v_ren["variable"] = to_v
    #print(v_to_agg_df)
    db_out = db_in.append(v_ren)
    # To make sure we use the pre-calculated version IF IT EXISTS,
    # we replace these values:

    return db_out

#def aggregate_variable(db_in, v_to_agg):
#    raise NotImplementedError("quantile handling wrong")
#    v_to_agg_df = (
#        db_in.filter(variable=v_to_agg, keep=False)
#        .filter(
#            variable="{}|*".format(v_to_agg),
#            level=0,  # make sure we don't pick up e.g. HFC23|50th Percentile by accident
#        )
#        .timeseries()
#    )
#    group_idx = list(set(v_to_agg_df.index.names) - {"variable"})
#    v_to_agg_df = v_to_agg_df.groupby(group_idx).sum().reset_index()
#    v_to_agg_df["variable"] = v_to_agg
#
#    db_out = db_in.append(v_to_agg_df)

#   return db_out





def get_protocol_vars(DATA_PROTOCOL,  sheet_name="variable_definitions"):
    protocol_variables = pd.read_excel(DATA_PROTOCOL, sheet_name=sheet_name)
    protocol_variables.columns = protocol_variables.columns.str.lower()
    protocol_variables.head()
    return protocol_variables

def get_protocol_scenarios(DATA_PROTOCOL, sheet_name='scenario_info'):
    protocol_scenarios = pd.read_excel(
        DATA_PROTOCOL, sheet_name=sheet_name, skip_rows=2
    )
    protocol_scenarios.columns = protocol_scenarios.columns.str.lower()
    return protocol_scenarios
    #protocol_scenarios.head()


def plot_available_out(db, variables, scenarios, figsize = [30, 30]):
    """
    Plots specified variables and scenarios to get overview over available data
    :param db: scmdata.dataframe input data
    :param variables: variable list to be plotted
    :param scenarios: scenario list to be plotted
    :param figsize: figure size
    :return:
    """
    fig, axs = plt.subplots(len(variables), len(scenarios), figsize=figsize, sharex = True)
    j = -1
    for var in variables:
        j += 1
        i = -1
        for scn in scenarios:
            i += 1
            ax = axs[j, i]
            _db = db.filter(variable=var, scenario=scenarios)
            _df = _db.filter(variable=var, scenario=scn)  # 'ssp119')#.timeseries()#[variables_erf[0]]
            __df = _df.timeseries().transpose()  # .plot(ax=ax)
            for model in __df.transpose().reset_index()[climatemodel]:
                # if model in __df:
                __df.xs(model, level=climatemodel, axis=1).plot(ax=ax)  # , label='sdf')
            ax.set_title(scn)
            ax.set_ylabel('W/m2, %s' % var.split('|')[-1])
            ax.set_xlim(['1850', '2100'])
            ax.legend(__df.transpose().reset_index()[climatemodel], frameon=False)
        # ax.set_ylim([-4,4])


def get_cmap_dic(keys, palette='colorblind'):
    cols = sns.color_palette(palette, n_colors=len(keys))
    colordic = {}  # 'NorESM1-LM':col}
    for model, col in zip(keys, cols):
        colordic[model] = col
    return colordic


def get_ls_dic(keys):
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot',(0, (1, 1))]
    odic = {}
    if len(keys) > len(linestyles):
        print('Warning: too many keys')
    for key, ls in zip(keys, linestyles):
        odic[key] = ls
    return odic

color_map_scenarios_base = {
    "ssp119": "AR6-SSP1-1.9",
    "ssp126": "AR6-SSP1-2.6",
    "ssp245": "AR6-SSP2-4.5",
    "ssp370": "AR6-SSP3-7.0",
    "ssp370-lowNTCF": "AR6-SSP3-LowNTCF",
    "ssp370-lowNTCF-aerchemmip": "AR6-SSP3-LowNTCF",
    "ssp434": "AR6-SSP4-3.4",
    "ssp460": "AR6-SSP4-6.0",
    "ssp585": "AR6-SSP5-8.5",
    "ssp534-over": "AR6-SSP5-3.4-OS",
    # "historical": "black",
    "rcp26": "AR5-RCP-2.6",
    "rcp45": "AR5-RCP-4.5",
    "rcp60": "AR5-RCP-6.0",
    "rcp85": "AR5-RCP-8.5",
    # "historical-cmip5": "tab:gray",
}
#color_map_scenarios_base = {
#    "ssp370-lowNTCF": "AR6-SSP3-LowNTCF",
#    "ssp370-lowNTCF-gidden": "tab:pink",
#    "ssp370-lowNTCF-aerchemmip": "AR6-SSP3-LowNTCF",
#}
scenario_list = ['ssp119', 'ssp126','ssp245','ssp370', 'ssp370-LowNTCF',
                 'ssp435','ssp460','ssp534os','ssp585']
ssp370low_nn = "ssp370-lowNTCF-aerchemmip"
ssp370low_on = 'ssp370-LowNTCF'

def trans_scen2plotlabel(label):
    if label==ssp370low_nn: return ssp370low_on
    else: return label

def get_scenario_c_dic(new=True):
    colormap_dic = {}

    if new:
        path_cf = BASE_DIR + '/misc/ssp_cat_2.txt'
        rgb_data_in_the_txt_file = np.loadtxt(path_cf)
        colormap_dic = {}
        for scn, col in zip(scenario_list, rgb_data_in_the_txt_file):
            colormap_dic[scn] = tuple([a/255. for a in col])#col/255.
        colormap_dic[ssp370low_nn] = colormap_dic[ssp370low_on]#'ssp370-lowNTCF']
        colormap_dic['historical'] = 'black'
        return  colormap_dic



    colormap_dic = {}
    ipccdic = pyam.plotting.PYAM_COLORS
    # print(ipccdic)
    for each in color_map_scenarios_base:
        key = color_map_scenarios_base[each]
        # print(key)
        colormap_dic[each] = ipccdic[color_map_scenarios_base[each]]
    colormap_dic['historical'] = 'black'
    # print(each)
    return colormap_dic

def get_scenario_ls_dic():
    c_dic = get_scenario_c_dic()
    ls_dic = {}
    for key in c_dic:
        ls_dic[key]='solid'
    ls_dic[ssp370low_nn] = 'dashed'
    return ls_dic



def prep_str_for_filename(ins):
    return (
        ins.replace("_", "-")
        .replace("|", "-")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def new_varname(var, nname):
    """
    var:str
        Old variable of format varname|bla|bla
    nname:str
        name for the resulting variable, based on var
    Returns
    -------
    new variable name with nname|bla|bla
    """
    return nname + '|' + '|'.join(var.split('|')[1:])