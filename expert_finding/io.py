import numpy as np
import pkg_resources
import logging

logger = logging.getLogger()


def load_dataset(dataset_name):
    npz_file = pkg_resources.resource_filename("expert_finding", 'resources/{0}.npz'.format(dataset_name))
    data = np.load(npz_file, allow_pickle=True)
    data_dict = dict()
    for k in data:
        if len(data[k].shape) == 0:
            data_dict[k] = data[k].flat[0]
        else:
            data_dict[k] = data[k]
        logger.debug(f"{k:>10} shape = {str(data_dict[k].shape):<20}  type = {str(type(data_dict[k])):<50}  dtype = {data_dict[k].dtype}")

    A_da = data_dict["A_da"]
    A_dd = data_dict["A_dd"]
    T = data_dict["T"]
    L_d = data_dict["L_d"]
    L_d_mask = data_dict["L_d_mask"]
    L_a = data_dict["L_a"]
    L_a_mask = data_dict["L_a_mask"]
    tags = data_dict["tags"]

    return A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags

def get_list_of_dataset_names():
    files_names =  pkg_resources.resource_listdir("expert_finding", 'resources')
    dataset_names = [f[:-4] for f in files_names if f[-4:] == ".npz"]
    return dataset_names