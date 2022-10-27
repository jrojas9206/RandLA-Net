import os 
import sys
import math
import glob  
import random
import argparse
import numpy as np 
from shutil import copy 
from multiprocessing import Process
#from sklearn.model_selection import train_test_split

def verify_folders(args):
    """
    verify that incoming and outgoing folders exist 
    args: must contain the reference to 'path2data' and 'output'
    """
    print("-Verifying defined input and output folders")
    if(not os.path.isdir(args.path2data)):
        print(" -[ERROR] The inpunt folder was not found: %s" %args.path2data)
        return -1
    if(not os.path.isdir(args.output)):
        print(" -[WARNING] The output folder doesn't exist and it is going to be created: %s" %(args.output))
        os.mkdir(args.output)
    print(" -Status: OK")
    return 0

def split_my_dataset(args):
    """
    """
    print("-Dataset Splitting")
    base_out_folders = ["train", "test"]
    list_files = glob.glob(os.path.join(args.path2data, "*.%s" %(args.format)))
    print(" -Found files: %i" %(len(list_files)))
    if(len(list_files)==0):
        print("  -No files were found on the input folder - path: %s - format of the files to look: %s" %(args.path2data, args.format))
        print("-EXIT")
        return 0
    random.seed(args.seed)
    random.shuffle(list_files)
    number_of_files_to_test = int(len(list_files)*args.p2test)
    test_list = list_files[:number_of_files_to_test]
    train_list = list_files[number_of_files_to_test:]
    to_split_data_ref = [train_list, test_list]
    print("   -Files to train [%i%%]: %i" %((100-(args.p2test*100)), len(train_list)))
    print("   -Files to test  [%i%%]: %i" %(((args.p2test*100)), len(test_list)))
    for a_folder, a_data in zip(base_out_folders, to_split_data_ref):
        path2out = os.path.join(args.output, a_folder)
        if(not os.path.isdir(path2out)):
            print("  -%s folder doesn't exist, it will be created: %s" %(a_folder, path2out))
            os.mkdir(path2out)
        if(args.cores==1):
            separate_data_into_folders( a_data, 
                                        path2out, 
                                        as_npy=args.saveAsnpy, 
                                        merge_instance=args.merge_instance, 
                                        label_column=args.label_column, 
                                        remove_label=args.remove_label)
        else:
            batches = split_on_batches_per_core(a_data, args.cores)
    return batches

def split_on_batches_per_core(data_list, number_of_cores):
    """
    """
    batch_size = math.floor(len(data_list)/(number_of_cores))
    print("Batch size: %.2f" %(batch_size))
    start = 0
    end = batch_size
    lst2return = []
    for a_core in range(number_of_cores):
        print("---Batch to core: %i------" %(a_core))
        tmp_list = []
        print("Start: %i" %(start))
        print("End: %i" %(end))
        for idx, list_idx in enumerate(range(start, end)):
            tmp_list.append(data_list[list_idx])
            if(idx == batch_size-1):
                start = list_idx
                end   = start+batch_size
                if(end-1 > len(data_list)):
                    end = len(data_list)
    
        lst2return.append(tmp_list)
    return lst2return

def separate_data_into_folders(data_list, output_path, ref=-1, as_npy=False, merge_instance=False, label_column=3, remove_label=False):
    """
    """
    for idx, a_file in enumerate(data_list, start=1):
        file_name_ref = os.path.split(a_file)[-1]
        if(as_npy or merge_instance or remove_label):
            print("-Copying[%i/%i]: %s" %(idx, len(data_list), "%s.%s" %(file_name_ref.split(".")[0], "npy" if as_npy else "txt")))
            new_pos = os.path.join(output_path, "%s.%s" %(file_name_ref.split(".")[0], "npy" if as_npy else "txt")) 
            try:
                actual_point_cloud = np.loadtxt(a_file)
            except:
                actual_point_cloud = np.loadtxt(a_file, delimiter=",")
            if(merge_instance and not remove_label):
                index2change = np.where(actual_point_cloud[:,label_column]>0)
                actual_point_cloud[index2change, label_column] = 1
            if(remove_label):
                actual_point_cloud = actual_point_cloud[:, 0:3]
            if(as_npy):
                np.save(new_pos, actual_point_cloud)
            else:
                np.savetxt(new_pos, actual_point_cloud, delimiter=",", fmt="%.6f")
        else:
            new_pos = os.path.join(output_path, file_name_ref)
            print("-Copying[%i/%i]: %s" %(idx, len(data_list), file_name_ref))
            copy(a_file, new_pos)
    return 0

def main():
    parser = argparse.ArgumentParser("Split dataset", "Verify the files in a folder and split them in train and test")
    parser.add_argument("path2data", type=str, help="Path to the folder")
    parser.add_argument("output", type=str, help="Path to the output folder")
    parser.add_argument("--format", type=str,help="Format of the files that have to be take into count, default:txt", default="txt")
    parser.add_argument("--p2test", type=float, help="Percentage to take as test, default:0.2", default=0.2)
    parser.add_argument("--saveAsnpy", help="Split the dataset and save the files as npy", action="store_true")
    parser.add_argument("--seed", type=int, help="Seed to shuffle the found data, default:42", default=42)
    parser.add_argument("--merge_instance", help="Take all the elements different to 0 and merge them in one single class", action="store_true")
    parser.add_argument("--label_column", type=int, help="Column with the instance annotations, default:3", default=3)
    parser.add_argument("--remove_label", help="Remove the label from the point clouds", action="store_true")
    parser.add_argument("--cores", type=int, help="Number of cores to achieve the desired task, if -1 is set all the cores are going to be used, default:1", default=1)
    args = parser.parse_args()

    folder_stat = verify_folders(args)
    # If the inpunt folder doesn't exist exit 
    if(folder_stat==-1):
        sys.exit()
    # Get the data and split it in train and test 
    split_my_dataset(args)

    return 0

if(__name__=="__main__"):
    #sys.exit(main())
    # Test 
    print("test")
    lst = glob.glob(os.path.join("/home/juan/Downloads/to_test_my_scripts/", "*.txt"))
    print(lst)
    if(len(lst)==0):
        print("error")
        sys.exit()
    print("len list: %i" %(len(lst)))
    batches = split_on_batches_per_core( lst, number_of_cores=3 )
    print(len(batches[0]))