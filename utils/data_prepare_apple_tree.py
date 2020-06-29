import numpy
import os
import glob
import pickle
import sys
import sklearn.neighbors

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


def convert_for_test(filename, output_dir, grid_size=0.005, synthetic=False):

    original_pc_folder = os.path.join(output_dir, 'test')
    if not os.path.exists(original_pc_folder):
        os.mkdir(original_pc_folder)

    sub_pc_folder = os.path.join(output_dir, 'input_{:.3f}'.format(grid_size))
    if not os.path.exists(sub_pc_folder):
        os.mkdir(sub_pc_folder)

    basename = os.path.basename(filename)[:-4]

    data = numpy.loadtxt(filename)

    points = data[:, 0:3].astype(numpy.float32)

    if synthetic:
        # TODO : hack must be remove
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)

    else:
        adr = normalize(data[:, 3:6]) * 255
        colors = adr.astype(numpy.uint8)

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


def convert_for_training(filename, output_dir, grid_size=0.005, synthetic=False):

    original_pc_folder = os.path.join(output_dir, 'training')
    if not os.path.exists(original_pc_folder):
        os.mkdir(original_pc_folder)

    sub_pc_folder = os.path.join(output_dir, 'input_{:.3f}'.format(grid_size))
    if not os.path.exists(sub_pc_folder):
        os.mkdir(sub_pc_folder)

    basename = os.path.basename(filename)[:-4]

    data = numpy.loadtxt(filename)

    points = data[:, 0:3].astype(numpy.float32)
    if synthetic:
        # TODO : hack must be remove
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)
        labels = data[:, 3].astype(numpy.uint8)

    else:
        adr = normalize(data[:, 3:6]) * 255
        colors = adr.astype(numpy.uint8)
        labels = data[:, 6].astype(numpy.uint8)

    field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'class']

    full_ply_path = os.path.join(original_pc_folder, basename + '.ply')

    #  Subsample to save space
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
    output_dir = "/gpfswork/rech/wwk/uqr22pt/data/apple_tree"
    grid_size = 0.001

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Generate Training data
    input_dir = "/gpfswork/rech/wwk/uqr22pt/data/apple_tree/field_afef_apple_tree_training/"
    training_filenames = glob.glob(input_dir + "*.txt")
    print(training_filenames, sep="\n")
    for filename in training_filenames:
        print(filename)
        convert_for_training(filename, output_dir, grid_size=grid_size)


    # Generate test data
    input_dir = "/gpfswork/rech/wwk/uqr22pt/field_afef_apple_tree_filtered/"
    training_basename = [os.path.basename(f) for f in training_filenames]

    test_filenames = glob.glob(input_dir + "*.txt")
    print(test_filenames, sep="\n")
    for filename in test_filenames:
        if os.path.basename(filename) in training_basename:
            print("not this one", filename)
            continue

        print(filename)
        convert_for_test(filename, output_dir, grid_size=grid_size)


def prepare_data_synthetic():
    output_dir = "/gpfswork/rech/wwk/uqr22pt/data/apple_tree_synthetic"
    grid_size = 0.001

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Generate Training data
    input_dir = "/gpfswork/rech/wwk/uqr22pt/synthetic_train/"
    training_filenames = glob.glob(input_dir + "*.txt")
    print(training_filenames, sep="\n")
    for filename in training_filenames:
        print(filename)
        convert_for_training(filename, output_dir, grid_size=grid_size, synthetic=True)

    # Generate test data
    input_dir = "/gpfswork/rech/wwk/uqr22pt/synthetic_test/"
    training_basename = [os.path.basename(f) for f in training_filenames]

    test_filenames = glob.glob(input_dir + "*.txt")
    print(test_filenames, sep="\n")
    for filename in test_filenames:
        if os.path.basename(filename) in training_basename:
            print("not this one", filename)
            continue

        print(filename)
        convert_for_test(filename, output_dir, grid_size=grid_size, synthetic=True)

if __name__ == "__main__":
    prepare_data_synthetic()
