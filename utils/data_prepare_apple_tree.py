import numpy
import os
import glob
import pickle
import sys
import sklearn.neighbors
import argparse 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
import helper_ply

from helper_ply import write_ply
from helper_tool import DataProcessing as DP


def normalize(adr):

    adr_min = numpy.array([7.00, -1.00, -25])
    adr_max = numpy.array([74.00, 15.00, 37])

    adr = (adr - adr_min) / (adr_max - adr_min) 

    return adr


def convert_for_test(filename, output_dir, grid_size=0.001, protocol="field"):

    original_pc_folder = os.path.join(output_dir, 'test')
    if not os.path.exists(original_pc_folder):
        os.mkdir(original_pc_folder)

    sub_pc_folder = os.path.join(output_dir, 'input_{:.3f}'.format(grid_size))
    if not os.path.exists(sub_pc_folder):
        os.mkdir(sub_pc_folder)

    basename = os.path.basename(filename)[:-4]

    data = numpy.loadtxt(filename)

    points = data[:, 0:3].astype(numpy.float32)

    if protocol == "synthetic" or protocol == "field_only_xyz":
        # TODO : hack must be remove
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)
    elif protocol == "field":
        adr = normalize(data[:, 3:-1]) * 255
        colors = adr.astype(numpy.uint8)
    else:
        exit("unknown protocol")

    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']

    #Save original
    full_ply_path = os.path.join(original_pc_folder, basename + '.ply')
    helper_ply.write_ply(full_ply_path, [points, colors], field_names)

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors = DP.grid_sub_sampling(points, colors, grid_size=grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = os.path.join(sub_pc_folder, basename + '.ply')
    helper_ply.write_ply(sub_ply_file, [sub_xyz, sub_colors], field_names)
    labels = numpy.zeros(data.shape[0], dtype=numpy.uint8)

    search_tree = sklearn.neighbors.KDTree(sub_xyz, leaf_size=50)
    kd_tree_file = os.path.join(sub_pc_folder, basename + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = numpy.squeeze(search_tree.query(points, return_distance=False))
    proj_idx = proj_idx.astype(numpy.int32)
    proj_save = os.path.join(sub_pc_folder, basename + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


def convert_for_training(filename, num_fold, output_dir, grid_size=0.001, protocol="field"):
    original_pc_folder = os.path.join(output_dir, 'training')
    if(num_fold is not None):
        fold_output_dir = os.path.join(output_dir, "fold_{}/".format(num_fold))
    else:
        fold_output_dir = original_pc_folder

    if not os.path.exists(fold_output_dir):
        os.mkdir(fold_output_dir)

    sub_pc_folder = os.path.join(output_dir, 'input_{:.3f}'.format(grid_size))
    if not os.path.exists(sub_pc_folder):
        os.mkdir(sub_pc_folder)

    basename = os.path.basename(filename)[:-4]

    data = numpy.loadtxt(filename)

    points = data[:, 0:3].astype(numpy.float32)
    if protocol == "synthetic":
        # TODO : hack must be remove
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)
        labels = data[:, -1].astype(numpy.uint8)
    elif protocol == "field_only_xyz":
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)
        labels = data[:, -1].astype(numpy.uint8)
    elif protocol == "field":
        adr = normalize(data[:, 3:-1]) * 255
        colors = adr.astype(numpy.uint8)
        labels = data[:, -1].astype(numpy.uint8)
    else:
        exit("unknown protocol")

    field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'class']

    full_ply_path = os.path.join(fold_output_dir, basename + '.ply')

    #  Subsample to save space
    # sub_points, sub_colors, sub_labels = DP.grid_sub_sampling(points, colors, labels, 0.01)
    #sub_labels = numpy.squeeze(sub_labels)
    # helper_ply.write_ply(full_ply_path, (sub_points, sub_colors, sub_labels), field_names)
    helper_ply.write_ply(full_ply_path, (points, colors, labels), field_names)

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(points, colors, labels, grid_size)
    sub_colors = sub_colors / 255.0
    sub_labels = numpy.squeeze(sub_labels)
    sub_ply_file = os.path.join(sub_pc_folder, basename + '.ply')
    helper_ply.write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], field_names)

    search_tree = sklearn.neighbors.KDTree(sub_xyz, leaf_size=50)
    kd_tree_file = os.path.join(sub_pc_folder, basename + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = numpy.squeeze(search_tree.query(points, return_distance=False))
    proj_idx = proj_idx.astype(numpy.int32)
    proj_save = os.path.join(sub_pc_folder, basename + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


def prepare_data_field():
    
    output_dir = "/gpfswork/rech/wwk/uqr22pt/data_RandLa-Net/apple_tree_field"
    grid_size = 0.001

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Generate Training data
    for i in range(1, 6):
        input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field/fold_{}/".format(i)

        training_filenames = glob.glob(input_dir + "*.txt")
        print(training_filenames, sep="\n")
        for filename in training_filenames:
            print(filename, flush=True)
            convert_for_training(filename, i, output_dir, outgrid_size=grid_size, protocol="field")

    # Generate test data
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field/test/"
    training_basename = [os.path.basename(f) for f in training_filenames]

    test_filenames = glob.glob(input_dir + "*.txt")
    print(test_filenames, sep="\n")
    for filename in test_filenames:
        if os.path.basename(filename) in training_basename:
            print("not this one", filename, flush=True)
            continue

        print(filename, flush=True)
        convert_for_test(filename, output_dir, grid_size=grid_size, protocol="field")


def prepare_data_field_only_xyz():
    output_dir = "/gpfswork/rech/wwk/uqr22pt/data_RandLa-Net/apple_tree_field_only_xyz_2"
    grid_size = 0.001

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Generate Training data
    for i in range(1, 6):
        input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field/fold_{}/".format(i)
        training_filenames = glob.glob(input_dir + "*.txt")
        print(training_filenames, sep="\n")
        for filename in training_filenames:
            print(filename, flush=True)
            convert_for_training(filename, i, output_dir, grid_size=grid_size, protocol="field_only_xyz")

    # Generate test data
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field/test/"
    training_basename = [os.path.basename(f) for f in training_filenames]

    test_filenames = glob.glob(input_dir + "*.txt")
    print(test_filenames, sep="\n")
    for filename in test_filenames:
        if os.path.basename(filename) in training_basename:
            print("not this one", filename, flush=True)
            continue

        print(filename, flush=True)
        convert_for_test(filename, output_dir, grid_size=grid_size, protocol="field_only_xyz")


def prepare_data_synthetic():
    output_dir = "/home/jprb/Documents/test_exp/low_res_sres_0004/all_splitted/manual_splitted/ntrain/prepare2randlanet/"
    grid_size = 0.001

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Generate Training data
    for i in range(1, 6):
        input_dir = "/home/jprb/Documents/test_exp/low_res_sres_0004/all_splitted/manual_splitted/ntrain/kfolds/fold_{}/".format(i)
        training_filenames = glob.glob(input_dir + "*.txt")
        print(training_filenames, sep="\n")
        for filename in training_filenames:
            print(filename, flush=True)
            convert_for_training(filename, i, output_dir, grid_size=grid_size, protocol="synthetic")

    # Generate test data
    input_dir = "/home/jprb/Documents/test_exp/low_res_sres_0004/all_splitted/manual_splitted/ntrain/kfolds/test/"
    training_basename = [os.path.basename(f) for f in training_filenames]

    test_filenames = glob.glob(input_dir + "*.txt")
    print(test_filenames, sep="\n")
    for filename in test_filenames:
        if os.path.basename(filename) in training_basename:
            print("not this one", filename, flush=True)
            continue

        print(filename, flush=True)
        convert_for_test(filename, output_dir, grid_size=grid_size, protocol="synthetic") 

def prepare_data_generic(path2data, path2output, grid_size=0.001, verbose=False, protocol="synthetic", dataset="train"):
    # Get the fail list 
    lst_fls = glob.glob(path2data + "*.txt")
    for idx, a_file in enumerate(lst_fls, start=1):
        if(verbose):
            print("-> Loading[%i/%i]: %s" %(len(lst_fls), idx, a_file))
        if(dataset=="train"):
            convert_for_training(a_file, None, path2output, protocol=protocol, grid_size=grid_size)
        elif(dataset=="test"):
            convert_for_test(a_file, path2output, grid_size=grid_size, protocol=protocol)
        else:
            raise ValueError("ERROR: Unknown option - %s" %dataset)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare the point clouds for the randlanet model")
    parser.add_argument("inputDir", type=str, help="Path to the directory with the point clouds")
    parser.add_argument("--outputDir", type=str, help="Path to write the new files", default="./output")
    parser.add_argument("--gridSize", type=str, help="--", default=0.001)
    parser.add_argument("--verbose", type=bool, help="Show verbose of each step", default=True)
    parser.add_argument("--ExpProtocol", type=str, help="Data over you apply the script, synthetic, field, field_only_xyz", default="synthetic")
    parser.add_argument("--datasetType", type=str, help="Part of the dataset that is going to be processed, train or test", default="train")
    args = parser.parse_args()
    print("-> Prepare data to randlanet model")
    prepare_data_generic(args.inputDir, args.outputDir, grid_size=args.gridSize, verbose=args.verbose, protocol=args.ExpProtocol, dataset=args.datasetType)
    #prepare_data_field()
    #prepare_data_field_only_xyz()
    #prepare_data_synthetic()