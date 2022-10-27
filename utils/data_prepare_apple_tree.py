from multiprocessing.sharedctypes import Value
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


def convert_for_test(filename, output_dir, grid_size=0.001, protocol="field", ColorColumns=[3,4,5], verbose=True):
    """
    Load a txt file that contain a Point Cloud and generted their related KDtree and 
    pyl file.

    :INPUT:
        filename: str, name of the txt file to process 
        output_dir: str, path to the folder where have to be saved the new files 
        grid_size: KDtree dimension reference to split the points 
        protocol: str, all the annotations are going to be remove, 
            - field_only_xyz: If the point cloud have the XYZ+Radiometric+Annotations
                              and in the output only want to be conserver the representation of XYZ
            - field: If the point clouds have XYZ+Radiometric+Annotations and in the output all the features
                     want to be conserved 
            - synthetic: If the point cloud only have the XYZ coordinates 
        ColorColumns: list, list of columns were are the colors or other features of the point clouds 
        verbose: bool, if true, print few message to know the development of each step 
    :OUTPUT:
        The fuction will create a folder called 'input_GRID_SIZE' were are going to be 
        the kdtree rerpesentation and differet pkl representation of the cloud.
        Apart from this folder called 'test' folder will be created, this folder contain the pyl
        files 
    """
    # Create the test folder 
    original_pc_folder = os.path.join(output_dir, 'test')
    if not os.path.exists(original_pc_folder):
        os.mkdir(original_pc_folder)
    # Create the base directory to save the KDTREE and the other representations
    sub_pc_folder = os.path.join(output_dir, 'input_{:.3f}'.format(grid_size))
    if not os.path.exists(sub_pc_folder):
        os.mkdir(sub_pc_folder)
    basename = os.path.basename(filename)[:-4]
    # LOAD FILE
    try:
        data = numpy.loadtxt(filename)
    except ValueError as err:
        data = numpy.loadtxt(filename, delimiter=",")
    
    # XYZ
    points = data[:, 0:3].astype(numpy.float32)
    # Create the variable to split the color or the radiometric features 
    if protocol == "synthetic" or protocol == "field_only_xyz":
        # TODO : hack must be remove
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)
    elif protocol == "field" or protocol=="synthetic_colors":
        if(len( ColorColumns ) >= 3 ):
            adr = normalize(data[:, ColorColumns]) * 255
        else: # Fit the positions 2,3 of features with Zeros
            cols2eval = data[ :, ColorColumns ]
            d2fit = numpy.zeros( (data.shape[0], 3-len(ColorColumns) ), dtype=numpy.uint8 )
            cols2eval = numpy.concatenate( ( cols2eval, d2fit ), axis=1 )
            adr = normalize( cols2eval ) * 255
        colors = adr.astype(numpy.uint8)
    else:
        exit("unknown protocol")

    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']

    if(verbose):
        print("  -> Filename: %s" %(filename))
        print("    -> Points shape: %s" %( str(points.shape) ) )
        print("    -> Features shape: %s" %( str(colors.shape) ))
        print("    -> ply order: %s" %( str( field_names ) ) )
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


def convert_for_training(filename, num_fold, output_dir, grid_size=0.001, protocol="field", columnOfLabels=6, ColorColumns=[3,4,5], verbose=True):
    """
    Load a txt file that contain a Point Cloud and generted their related KDtree and 
    pyl file. 

    :INPUT:
        filename: str, name of the txt file to process 
        output_dir: str, path to the folder where have to be saved the new files 
        grid_size: KDtree dimension reference to split the points 
        protocol: str, all the annotations are going to be remove, 
            - field_only_xyz: If the point cloud have the XYZ+Radiometric+Annotations
                              and in the output only want to be conserver the representation of XYZ and the annotations
            - field: If the point clouds have XYZ+Radiometric+Annotations and in the output all the features and annotations
                     want to be conserved 
            - synthetic: If the point cloud only have the XYZ coordinates 
        columnOfLabels: Column where are the annotations of the points
        ColorColumns: list, list of the columns that have the colors or other features of the point clouds 
    :OUTPUT:
        The fuction will create a folder called 'input_GRID_SIZE' were are going to be 
        the kdtree rerpesentation and differet pkl representation of the cloud.
        Apart from this folder called 'test' folder will be created, this folder contain the pyl
        files 
    """

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

    try:
        data = numpy.loadtxt(filename)
    except ValueError as err:
        data = numpy.loadtxt(filename, delimiter=",")

    points = data[:, 0:3].astype(numpy.float32)
    labels = data[:, columnOfLabels].astype(numpy.uint8)

    if(protocol == "synthetic"):
        # TODO : hack must be remove
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)
        adr = colors
    elif(protocol == "field_only_xyz"):
        colors = numpy.zeros((data.shape[0], 3), dtype=numpy.uint8)
        adr = colors
    elif(protocol == "field" or protocol=="synthetic_colors"):
        if(len( ColorColumns ) >= 3 ):
            adr = normalize(data[:, ColorColumns]) * 255
        else: # Fit the positions 2,3 of features with Zeros
            cols2eval = data[ :, ColorColumns ]
            d2fit = numpy.zeros( (data.shape[0], 3-len(ColorColumns) ), dtype=numpy.uint8 )
            cols2eval = numpy.concatenate( ( cols2eval, d2fit ), axis=1 )
            adr = normalize( cols2eval ) * 255
    else:
        exit("unknown protocol")
        
    colors = adr.astype( numpy.uint8 )
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'class']

    if(verbose):
        print("  -> Filename: %s" %(filename))
        print("   -> Points shape: %s" %( str(points.shape) ) )
        print("   -> Features shape: %s" %( str(colors.shape) ) )
        print("   -> Found lables: %s" %( str( numpy.unique( labels ) ) ) )
        print("      -> %i" %( int(labels.shape[0]) ))
        print("   -> ply order: %s" %( str( field_names ) ) )
        print("   -> Feature columns: %s" %( str( ColorColumns ) ) )

    full_ply_path = os.path.join(fold_output_dir, basename + '.ply')

    #  Subsample to save space
    # sub_points, sub_colors, sub_labels = DP.grid_sub_sampling(points, colors, labels, 0.01)
    #sub_labels = numpy.squeeze(sub_labels)
    # helper_ply.write_ply(full_ply_path, (sub_points, sub_colors, sub_labels), field_names)
    helper_ply.write_ply(full_ply_path, (points, colors, labels), field_names)

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(points, features=colors, labels=labels, grid_size=grid_size)
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

def prepare_data_generic(path2data, path2output, grid_size=0.001, verbose=False, protocol="synthetic", dataset="train", annCol=3, colorCol=[3,4,5]):
    """
    Prepare the data for the RandLA-Net Model
    :INPUT:
        path2data: str, Path to the folder with the point clouds on *.txt format 
        path2output: str, Path to the folder where have to be written the processed files [*.ply, *.pkl]
        grid_size: float, Size of each block of the grid , default, 0.001
        verbose: bool, if true print messages related to the evolution of the script 
        protocol: str, Name of the protocol to prepare, synthetic/field/field_only_xyz/synthetic_colors
        dataset: str, Type of the dataset to prepare, test or train. If test the annotatios are remove 
        annCol: int, Index of the column where are located the annotations 
        colorCol: list, List of the columns with some features of the point cloud, default [3,4,5]
    """
    # Get the fail list 
    lst_fls = glob.glob(path2data + "*.txt")
    for idx, a_file in enumerate(lst_fls, start=1):
        if(verbose):
            print("-> Loading[%i/%i]: %s" %(len(lst_fls), idx, a_file))
        if(dataset=="train"):
            convert_for_training(a_file, None, path2output, protocol=protocol, grid_size=grid_size, columnOfLabels=annCol, ColorColumns=colorCol)
        elif(dataset=="test"):
            convert_for_test(a_file, path2output, grid_size=grid_size, protocol=protocol, ColorColumns=colorCol)
        else:
            raise ValueError("ERROR: Unknown option - %s" %dataset)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare the point clouds for the randlanet model")
    parser.add_argument("inputDir", type=str, help="Path to the directory with the point clouds")
    parser.add_argument("--outputDir", type=str, help="Path to write the new files,  default=./output", default="./output")
    parser.add_argument("--gridSize", type=str, help="Grid size to split the point cloud, default=0.001", default=0.001)
    parser.add_argument("--verbose", help="Show verbose of each step", action="store_true", default=False)
    parser.add_argument("--ExpProtocol", type=str, help="Data over you apply the script, synthetic, field, field_only_xyz, default=synthetic", default="synthetic")
    parser.add_argument("--datasetType", type=str, help="Part of the dataset that is going to be processed, train or test. Default=train", default="train")
    parser.add_argument("--annColumn", type=int, help="Column with the annotated labels, default=6", default=6)
    parser.add_argument("--featureCols", type=str, help="Column Id of the point features, default = [3,4,5]", default="3,4,5")
    args = parser.parse_args()
    print("-> Prepare data to randlanet model")
    feature_col = args.featureCols.split(",")
    feature_col = [ int(i) for i in feature_col ]
    prepare_data_generic(args.inputDir, args.outputDir, grid_size=args.gridSize, verbose=args.verbose, protocol=args.ExpProtocol, dataset=args.datasetType, annCol=args.annColumn, colorCol=feature_col)