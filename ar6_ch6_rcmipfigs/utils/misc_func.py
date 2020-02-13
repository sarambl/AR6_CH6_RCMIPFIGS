import logging
import os

import pandas as pd
from scmdata import ScmDataFrame

climatemodel = 'climatemodel'
logger = logging.getLogger()


def aggregate_variable(db_in, v_to_agg, cmodel, remove_quantiles=True):
    """
    Based on Zebedee Nicholls  (zebedee.nicholls@climate-energy-college.org) code https://gitlab.com/rcmip/rcmip
    Aggregates variables for specified model. E.g.
    v_to_agg='Effective radiative forcing|Anthropogenic' will sum up all subcategories of
    'Effective radiative forcing|Anthropogenic'

    :type db_in: ScmDataFrame
    :param db_in: data to aggregate
    :param v_to_agg: variable to aggregate, str.
    :param cmodel: climatemodel
    :param remove_quantiles: True
    :return:
    """
    if not remove_quantiles:
        raise NotImplementedError("quantile handling wrong")
    # remove quantiles (so they are not aggregated
    _db = db_in.filter(variable='*quantile', keep=False).filter(climatemodel=cmodel)
    # Check if variable already there -- if so, keeps original and does not overwrite
    for cm in _db[climatemodel].unique():
        if len(_db.filter(variable=v_to_agg, climatemodel=cm)['scenario'].unique()) > 0:
            print('Model %s already has variable, returns unchanged' % cmodel)
            return db_in

    # The variables to aggregate:
    v_to_agg_df = (
        _db.filter(variable=v_to_agg, keep=False)  # remove v_to_agg
            .filter(
            variable="{}|*".format(v_to_agg),  # pick out subgroups.
            level=0,  # make sure we don't pick up e.g. HFC23|50th Percentile by accident
        )
            .timeseries()
    )
    # Check if variables found:
    if len(v_to_agg_df) == 0:
        print('No variables to aggregate found in model %s. Returns unchanged' % cmodel)
        return db_in

    # Group by index except 'variable'
    group_idx = list(set(v_to_agg_df.index.names) - {"variable"})
    groups = v_to_agg_df.groupby(group_idx).groups

    for g in groups.keys():
        print('Aggragating for model %s:*' % cmodel)
        print('Group:')
        print(g)
        print('Aggregates:')
        print(list(groups[g]))
    # Finally, sum up values in each group and set values in db_out
    v_to_agg_df = v_to_agg_df.groupby(group_idx).sum().reset_index()
    v_to_agg_df["variable"] = v_to_agg
    db_out = db_in.append(v_to_agg_df)

    return db_out


def fix_BC_name(db_in,
                from_v='Effective Radiative Forcing|Anthropogenic|Albedo Change|Other|Deposition of Black Carbon on Snow',
                to_v='Effective Radiative Forcing|Anthropogenic|Other|BC on Snow'):
    """
    Changes variable name in db
    :param db_in:
    :param from_v: Original name
    :param to_v: output name
    :return: db with from_v changed to to_v
    """
    # Convert to dataframe:
    db = db_in.timeseries().reset_index()
    # Replace name:
    db["variable"] = db["variable"].apply(lambda x: to_v if x == from_v else x)
    # convert back to ScmDataFrame
    db = ScmDataFrame(db)
    return db


def get_protocol_vars(DATA_PROTOCOL, sheet_name="variable_definitions"):
    """
    Based on Zebedee Nicholls  (zebedee.nicholls@climate-energy-college.org) code https://gitlab.com/rcmip/rcmip
    """
    protocol_variables = pd.read_excel(DATA_PROTOCOL, sheet_name=sheet_name)
    protocol_variables.columns = protocol_variables.columns.str.lower()
    protocol_variables.head()
    return protocol_variables


def get_protocol_scenarios(DATA_PROTOCOL, sheet_name='scenario_info'):
    """
    Based on Zebedee Nicholls  (zebedee.nicholls@climate-energy-college.org) code https://gitlab.com/rcmip/rcmip
    """
    protocol_scenarios = pd.read_excel(
        DATA_PROTOCOL, sheet_name=sheet_name, skip_rows=2
    )
    protocol_scenarios.columns = protocol_scenarios.columns.str.lower()
    return protocol_scenarios


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


def make_folders(path):
    """
    Takes path and creates to folders
    :param path: Path you want to create (if not already existant)
    :return: nothing

    Example (want to make folders for placing file myfile.png:
    >>> path='my/folders/myfile.png'
    >>> make_folders(path)
    """
    path = extract_path_from_filepath(path)
    split_path = path.split('/')
    if path[0] == '/':

        path_inc = '/'
    else:
        path_inc = ''
    for ii in range(0, len(split_path)):
        # if ii==0: path_inc=path_inc+split_path[ii]
        path_inc = path_inc + split_path[ii]
        if not os.path.exists(path_inc):
            os.makedirs(path_inc)
        path_inc = path_inc + '/'

    return


def extract_path_from_filepath(file_path):
    """
    ex: 'folder/to/file.txt' returns 'folder/to/'
    :param file_path:
    :return:
    """

    st_ind = file_path.rfind('/')
    foldern = file_path[0:st_ind] + '/'
    return foldern
