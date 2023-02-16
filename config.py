 # config.py ---

import argparse

def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_dump_prefix", type=str, default="E:/NM-Net-Initial/datasets", help=""
    "prefix for the dump folder locations")
data_arg.add_argument(
    "--data_tr", type=str, default="south", help="")  #person.south.gerrard.graham   "graham"
data_arg.add_argument(
    "--data_crop_center", type=str2bool, default=False, help=""
    "whether to crop center of the image "
    "to match the expected input for methods that expect a square input")
# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("obj")
obj_arg.add_argument(
    "--obj_num_kp", type=int, default=2000, help=""
    "number of keypoints per image")
obj_arg.add_argument(
    "--obj_geod_type", type=str, default="episym",
    choices=["sampson", "episqr", "episym"], help=""
    "type of geodesic distance")
obj_arg.add_argument(
    "--obj_geod_th", type=float, default=1e-4, help=""  
    "theshold for the good geodesic distance")
obj_arg.add_argument(
    "--obj_num_nn", type=int, default=1, help=""
    "number of nearest neighbors in terms of descriptor "
    "distance that are considered when generating the "
    "distance matrix")
# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--train_batch_size", type=int, default=1, help=""        # 10    1  16 32 16
    "batch size")
train_arg.add_argument(
    "--knn_num", type=int, default=8, help=""               # 8
    "the number of neighbors")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-3, help=""       # 0.00001       # 0.000000000000001, help=""      # 0.001 1e-3      0.0080
    "learning rate")
train_arg.add_argument(
    "--res_dir", type=str, default="../logs", help=""
    "base directory for results")
train_arg.add_argument('--epochs', type=int, default=15, metavar='N',help=     # 8    50 4     150
    'number of episode to train ')
# -----------------------------------------------------------------------------
# Others
others_arg = add_argument_group("Others")
others_arg.add_argument('--restore', type=bool, default=False, help=     # False
    'if use model to test datasets ')
others_arg.add_argument('--use_which_network', type=int, default=0, help=      # 0
    '0 multimodal_network 1 nm-net 2 cne 3 oanet 4 acne')

def setup_dataset(dataset_name):
    """Expands dataset name and directories properly"""

    # Use only the first one for dump
    dataset_name = dataset_name.split(".")[0]
    data_dir = "../datasets/"

    # Expand the abbreviations that we use to actual folder names
    if "science-narrow" == dataset_name:
        # Load the data
        data_dir += "science-narrow/"
        geom_type = "Calibration"
        vis_th = 100
    elif "lib-narrow" == dataset_name:
        # Load the data
        data_dir += "lib-narrow/"
        geom_type = "Calibration"
        vis_th = 100
    elif "lib-wide" == dataset_name:
        # Load the data
        data_dir += "lib-wide/"
        geom_type = "Calibration"
        vis_th = 100
    elif "science-wide" == dataset_name:
        # Load the data
        data_dir += "science-wide/"
        geom_type = "Calibration"
        vis_th = 100
    elif "mao-wide" == dataset_name:
        # Load the data
        data_dir += "mao-wide/"
        geom_type = "Calibration"
        vis_th = 100
    elif "main-wide" == dataset_name:
        # Load the data
        data_dir += "main-wide/"
        geom_type = "Calibration"
        vis_th = 100
    elif "mao-narrow" == dataset_name:
        # Load the data
        data_dir += "mao-narrow/"
        geom_type = "Calibration"
        vis_th = 100
    elif "main-narrow" == dataset_name:
        # Load the data
        data_dir += "main-narrow/"
        geom_type = "Calibration"
        vis_th = 100
    elif "person" == dataset_name:
        # Load the data
        data_dir += "COLMAP/person/"
        geom_type = "Calibration"
        vis_th = 100
    elif "graham" == dataset_name:
        # Load the data
        data_dir += "COLMAP/graham/"
        geom_type = "Calibration"
        vis_th = 100
    elif "gerrard" == dataset_name:
        # Load the data
        data_dir += "COLMAP/gerrard/"
        geom_type = "Calibration"
        vis_th = 100
    elif "south" == dataset_name:
        # Load the data
        data_dir += "south/"
        geom_type = "Calibration"
        vis_th = 100

    return data_dir, geom_type, vis_th


def get_config():
    config, unparsed = parser.parse_known_args()

    # Setup the dataset related things
    _mode = "tr"
    data_dir, geom_type, vis_th = setup_dataset(
        getattr(config, "data_" + _mode))
    setattr(config, "data_dir_" + _mode, data_dir)
    setattr(config, "data_geom_type_" + _mode, geom_type)
    setattr(config, "data_vis_th_" + _mode, vis_th)

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
