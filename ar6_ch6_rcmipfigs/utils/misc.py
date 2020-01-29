import os

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
    if (path[0] == '/'):

        path_inc = '/'
    else:
        path_inc = ''
    for ii in range(0,len(split_path)):
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

    st_ind=file_path.rfind('/')
    foldern = file_path[0:st_ind]+'/'
    return foldern
