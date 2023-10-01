import os 
import sys 
import glob 
import pickle
import sklearn
import argparse 
import numpy as np 
import sklearn.neighbors
from functools import partial
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
import helper_ply

from helper_ply import write_ply
from helper_tool import DataProcessing as DP

def normalize(radiometric, selection=[1,1,1]):
    """
        Normalize the Radiometric features of the LiDAR scan 
        :param radiometric: numpy array, Radiometric features
        :param selection: list, List of lenght 3 with the selected columns to work, 
                          a list with 0 and 1 is expected [1,0,1] = [amplitude[on], deviation[off], reflectance[on]]
        :return: numpy array
    """
    if(len(selection)!=3):
        raise ValueError("It is expected a list of length 3")
    # adr = amplitude deviation reflectance
    adr_min = np.array([standarized*active for standarized, active in zip([7.00, -1.00, -25], selection)])
    adr_max = np.array([standarized*active for standarized, active in zip([74.00, 15.00, 37], selection)])
    #
    if(sum(selection)==3):
        return (radiometric-adr_min)/(adr_max-adr_min)
    elif(sum(selection)==0):
        return radiometric
    else: # Ensure there is no zero divition due to the column selection 
        matrix2return = np.array([])
        for a_column in range(len(selection)):
            if selection[a_column]==0:
                if a_column == 0:
                    matrix2return = np.zeros((radiometric.shape[0]), dtype=np.uint8).reshape(-1,1)
                else:
                    matrix2return = np.concatenate([matrix2return, np.zeros((radiometric.shape[0], 1), dtype=np.uint8)], axis=1) # 1
                continue
            normalized_column = (radiometric[:,a_column ]-adr_min[a_column])/((adr_max[a_column]-adr_min[a_column]))
            if a_column == 0:
                matrix2return = normalized_column.reshape(-1,1)
            else:
                matrix2return = np.concatenate([matrix2return, normalized_column.reshape(-1,1)], axis=1) # 2
        return matrix2return

def verify_main_folders(args):
    """
        Verify that the input are output folders are well define 

        :param args: object ArgumentParser, The object must have the attributes
                     - path2data [input folder], 
                     - path2out [output folder], 
                     - verbose: if verbose is set to true, few messages will be 
                                print
        :return: int
            O if all is fine
           -1 if there is error
    """
    if(args.verbose):
        print(" -o Verify folders o-")
    if(not os.path.isdir(args.path2data)):
        if args.verbose:
            print ("   -x The defined input folder doesn't exist")
        return -1
    if args.verbose:
        print("  -> Input folder exist")
    list_files = glob.glob(os.path.join(args.path2data, "*.%s" %(args.fileExtension)))
    if(len(list_files)<1):
        if args.verbose:
            print("   -x No files were found in the input folder")
        return -1
    if args.verbose:
        print("   -> Found files: %i" %(len(list_files)))
    if(not os.path.isdir(args.path2out)):
        os.mkdir(args.path2out)
        if args.verbose:    
            print("   -x Output folder will be created")
    if args.verbose:
        print("  -> Output folder OK")
    return 0

def load_pointcloud(ifileName):
    """
        Load pointcloud files
        :param ifileName: str, file to load (Only txt files for the moment)
        :return: numy array 
    """
    try:
        pointcloud = np.loadtxt(ifileName)
    except ValueError:
        pointcloud = np.loadtxt(ifileName, delimiter=",")
    return pointcloud.astype(np.float32)

def prepare_point_clouds(filesList, path2out="out", gridsize=0.001, 
                         verbose=True, train=True, experiment="field",
                         useReflectance=[1,1,1], annotation_available=True,
                         debug=False):
    """
        Load the point clouds and generate their KDtree and pyl versions

        :param fileList: list, List with the path to the point clouds 
        :param path2out: str, path to write the outputs
        :param gridsize: float, Spatial grid to divide the point cloud 
        :param verbose: bool, If true several message are going to be printed 
        :param experiment: str, Define the type of data and processing to follow.
                                options are: 'field' - For LiDAR point clouds - Return XYZ + Radiometric features + Annotation [If train is true]
                                'field_only_xyz' - For LiDAR point clouds - return the processing of XYZ + Annotations [If train is ture]
                                'synthetic' - For simulated LiDAR Point clouds - Return the processing of XYZ + Annatotions [if train is true]
        :param annotation_available: bool, If true labels are take into account for the processing
        :param debug: bool, if True, a dictionary with all the possible outputs will be returned for verification 
        :return: int
            O if all is fine 
           -1 if there is a problem 
        :NOTE:
            the following folder will be created:
                - input_x.xxx
                - training || test 
        :NOTE:
            - It is assumed that the first 3 columns of the files are the 
            XYZ coordinates of the point cloud
            - It is assumed that the last column it is the annotation column
    """
    if verbose:
        print(" -o prepare_point_clouds o-")
    # Folders to create
    subfolder_extra_files = os.path.join(path2out, "input_%.3f" %(gridsize))
    subfolder_train_test = os.path.join(path2out, "training" if train else "test")
    if not os.path.isdir(subfolder_extra_files):
        print("  -> Subfolder for datastrcture is going to be created: %s" %(subfolder_extra_files))
        try:
            os.mkdir(subfolder_extra_files)
        except FileExistsError:
            print("-x Folder already exist %s" %(subfolder_extra_files))
    else:
        print("  -> Subfolder for datastructure already exist")
    if not os.path.isdir(subfolder_train_test):
        print("  -> Folder for the pointclouds in pyl format is going to be created")
        try:
            os.mkdir(subfolder_train_test)
        except FileExistsError:
            print("-x Folder already exist %s" %(subfolder_train_test))
    else:
        print("  -> Subfolder for the pointclouds on pyl format already exist")
    # Loading and processing point clouds 
    for idx, a_file in enumerate(filesList, start=1): 
        if verbose:
            print("   -> Processing[%i/%i]" %(idx, len(filesList)))
        pointcloud_name = os.path.split(a_file)[-1]
        pointcloud_ply = "%s.ply" %(pointcloud_name.split('.')[0])
        pointcloud = load_pointcloud(a_file)
        points = pointcloud[:, 0:3] # XYZ
        # Set labels for training and test 
        if train and annotation_available:
            lables = pointcloud[:, -1].astype(np.uint8) # Points annotation 
        else:
            lables = np.zeros(pointcloud.shape[0], dtype=np.int8) # No annotations
        # Normalize and set colors / radiometric
        if experiment == "field_only_xyz" or experiment == "synthetic":
            color2ply = np.zeros((points.shape[0], 3), dtype=np.uint8)
        elif experiment == "field":
            if annotation_available and train:
                color2ply = (normalize(pointcloud[:, 3:pointcloud.shape[1]-1],
                                  selection=useReflectance)*255).astype(np.uint8) 
            else:
                if annotation_available:
                    color2ply = (normalize(pointcloud[:, 3:pointcloud.shape[1]-1],
                                  selection=useReflectance)*255).astype(np.uint8)
                else:
                    color2ply = (normalize(pointcloud[:, 3:pointcloud.shape[1]],
                                  selection=useReflectance)*255).astype(np.uint8)
        else:
            raise ValueError("Uknown experiment options: %s" %(experiment))
        # Ply standard column names
        ply_fields = ['x', 'y', 'z', 'red', 'green', 'blue']
        if verbose:
            print("    -> File name: %s" %(pointcloud_name))
            print("      -> Shape: %s" %(str(pointcloud.shape)))
            print("      -> Feature shape: %s" %(str(color2ply.shape)))
            print("        -> red[length unique values - values]: %i - %s" %(len(np.unique(color2ply[:,0])), str(np.unique(color2ply[:,0])) if len(np.unique(color2ply[:,0]))<4 else "..."))
            print("        -> green[length unique values - values]: %i - %s" %(len(np.unique(color2ply[:,1])), str(np.unique(color2ply[:,1])) if len(np.unique(color2ply[:,1]))<4 else "..."))
            print("        -> blue[length unique values - values]: %i - %s" %(len(np.unique(color2ply[:,2])), str(np.unique(color2ply[:,2])) if len(np.unique(color2ply[:,2]))<4 else "..."))
            if(annotation_available):
                print("        -> Label[lenght - values]: %i - %s" %(lables.shape[0], str(np.unique(lables))))
        # Write PLY files
        folder2ply = os.path.join(subfolder_train_test, pointcloud_ply)
        helper_ply.write_ply(folder2ply, [points, color2ply], ply_fields)
        # Subsample pointcloud and write new PLY file 
        folder2ply_subsampled_pointcloud = os.path.join(subfolder_extra_files, pointcloud_ply)
        subsampled_points, subsampled_colors = DP.grid_sub_sampling(points, # Point cloud XYZ
                                                                    color2ply,  # Colors | Reflectances
                                                                    grid_size=gridsize) # Voxel size 
        subsampled_colors = subsampled_colors/255.0
        if verbose:
            print("    -> Pointcloud subsampled: %s" %(str(subsampled_points.shape)))
            print("    -> Colors subsampled: %s" %(str(subsampled_colors.shape)))
        helper_ply.write_ply(folder2ply_subsampled_pointcloud,
                             [subsampled_points, subsampled_colors],
                             ply_fields)
        # Get and write KDtree
        folder2kdtree = os.path.join(subfolder_extra_files, "%s_KDTree.pkl" %(pointcloud_name.split('.')[0]))
        search_tree = sklearn.neighbors.KDTree(subsampled_points,
                                               leaf_size=50)
        with open(folder2kdtree, "wb") as f:
            pickle.dump(search_tree, f)
        # 
        folder2project_pointcloud = os.path.join(subfolder_extra_files, "%s_proj.pkl" %(pointcloud_name.split('.')[0]))
        projected_pointcloud_idx = np.squeeze(search_tree.query(points, return_distance=False))
        projected_pointcloud_idx = projected_pointcloud_idx.astype(np.int32)
        with open(folder2project_pointcloud, "wb") as f:
            pickle.dump([projected_pointcloud_idx, lables], f)

def main():
    parser = argparse.ArgumentParser("Prepare apple tree LiDAR point cloud to RandLA-NET")
    parser.add_argument("path2data", type=str, help="Path to the point clouds to process")
    parser.add_argument("path2out", type=str, help="Path to write the desired data")
    parser.add_argument("--gridSize", type=float, help="Grid size to take into account, default:0.001", default=0.001)
    parser.add_argument("--experiment", type=str, help="Define 'field' if you want process LiDAR data with the radiometric features\
                        'field_only_xyz' if you want to process the XYZ coordinates from LiDAR,\
                        , and 'synthetic' if you want to process files only with the XYZ coordinates.\
                        This script expect that the data has the size of Nx7 for field data and Nx4 for synthetic. default: field", default="field")
    parser.add_argument("--test", help="Data for test. If it is not set, the default is data to train", action="store_true")  
    parser.add_argument("--runUnitTest", help="Verify that each method is doind the correct work, NOTE: Unit test force the run in 1 core", action="store_true")
    parser.add_argument("--fileExtension", type=str, help="File extension, default:txt", default="txt")
    parser.add_argument("--verbose", help="Print messages", action="store_true")
    parser.add_argument("--not_select_reflectance", help="Select the reflectance column", action="store_true")
    parser.add_argument("--not_select_amplitude", help="Select the amplitude column", action="store_true")
    parser.add_argument("--not_select_deviation", help="Select the deviation column", action="store_true")
    parser.add_argument("--no_annotations_available", help="The data doesn't have annotation column", action="store_true")
    parser.add_argument("--cores", type=int, help="Number of cores to use in the task, default:1", default=1)
    args = parser.parse_args()

    folders_status = verify_main_folders(args)
    activate_reflectance = [int(not args.not_select_amplitude), 
                            int(not args.not_select_deviation), 
                            int(not args.not_select_reflectance)]
    if(folders_status == -1):
        print("[ERROR] There is some problem in the defined paths")
        sys.exit()

    pointcloud_file_list = glob.glob(os.path.join(args.path2data, "*.%s" %(args.fileExtension)))

    if args.cores == 1 or args.runUnitTest: 
        prepare_point_clouds(filesList = pointcloud_file_list,
                            path2out = args.path2out,
                            gridsize = args.gridSize,
                            verbose = args.verbose,
                            train = not args.test,
                            experiment = args.experiment,
                            useReflectance = activate_reflectance,
                            annotation_available = not args.no_annotations_available,  
                            debug=args.runUnitTest)
    elif args.cores > 1:
        chunks = np.array_split(pointcloud_file_list, args.cores)
        with Pool(args.cores) as p:
            p.map(partial(prepare_point_clouds,
                            path2out = args.path2out,
                            gridsize = args.gridSize,
                            verbose = args.verbose,
                            train = not args.test,
                            experiment = args.experiment,
                            useReflectance = activate_reflectance,
                            annotation_available = not args.no_annotations_available,), chunks)
    if args.verbose:
        print(" --o EXIT o--")
    return 0

if __name__ == "__main__":
    sys.exit(main())
