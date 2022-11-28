import os 
import sys
import glob 
import json
import argparse
import numpy as np 
from math import floor
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Manager
from sklearn.neighbors import KDTree

def verify_paths(args):
    """
    Verify that the input and output folders exist 

    :INPUT:
        args: parser args got from the arparse package, 
              NOTE: the expected variables are args.path2pointclouds  
                    and args.path2out 
    :OUT:
        int, -1 if one or both of the folders doesn't exist 
              0 if the the both folders exist 
    """
    if(not os.path.isdir(Path(args.path2pointclouds).resolve())):
        print(" ->[ERROR] Input path was not found")
        return -1
    if(not os.path.isdir(args.path2out)):
        print(" ->[WARNING] Output folder was not found, it is going to be created")
        os.mkdir(Path(args.path2out).resolve())
        print("   -> Stat: OK")
    return 0

def get_point_cloud_density(pc, radius=0.1):
    """
    Get the point density of a point cloud from several 
    density measurements arround the point cloud  
    
    Npoints/R**2? -- must be Npoints/ 4/3*pi*R**3?

    :INPUT:
        pc: numpy array (N,3)
        radius: Distance to evaluate the nearest points, sphere radius 
    :OUTPUT:
        list of float32
    """
    # Get the KDtree representation of the point cloud 
    # and from it begin to estimate the densities 
    # base on the number of points inside the 
    # evaluate sphere 

    kdt_rep = KDTree(pc, leaf_size=2)
    nPts = kdt_rep.query_radius(pc, r=radius, count_only=True)
    lstDens = nPts/( radius**2 )
    return lstDens

def get_pointcloud_general_characteristics(lst_files, annColumn=3, radius=0.1, idx_core=-1, only_npoints=False):
    """
    Get the density and the number of points of the referenced point clouds 

    :INPUT:
        lst_files: list, List with all the path to the files that must be processed 
        annColumn: int,  Integer with the column number where are located the point labels 
        radius   : float, float that represent the radius of the sphere used to calculate the point densities 
        idx_core : int, integer used to know if the method have been called from a threat, -1 means single threat 
                        in this case the model return a dictioray, if the idx_core > 1 it will write a set of json 
                        files that contain the result of the analysis of the point clouds 
    :OUT:
        dict  if idx_core is equal to -1
        write a file    if idx_core is bigger than 1
    """
    dic2return = {}
    for idx, a_file in enumerate(lst_files, start=1):
        if(idx_core == -1):
            print("-> Processing[%i/%i]:%s" %(idx, len(lst_files), os.path.split(a_file)[-1]))
        else:
            print("-> Processing[%i/%i][core:%i]:%s" %(idx, len(lst_files), idx_core, os.path.split(a_file)[-1]))
        try:
            actual_pointcloud = np.loadtxt(a_file)
        except ValueError as err:
            print("    -> ERR: %s" %(err))
            actual_pointcloud = np.loadtxt(a_file, delimiter=",")
        print("  -> Point cloud shape: %s" %(str(actual_pointcloud.shape)))
        if(actual_pointcloud.shape[1]>3):
            print("  -> Found labels: %i" %(len(np.unique(actual_pointcloud[:,annColumn])))) 
        if("names" not in dic2return.keys()):
            dic2return["names"] = [os.path.split(a_file)[-1].split(".")[0]]
            dic2return["number_of_points"] = [actual_pointcloud.shape[0]]
            if(not only_npoints):
                dic2return["avg_densities"] = [np.mean(np.array(get_point_cloud_density(actual_pointcloud[:,0:3], radius=radius)))]
        else:
            
            dic2return["names"].append(os.path.split(a_file)[-1].split(".")[0])
            dic2return["number_of_points"].append(actual_pointcloud.shape[0])
            if(not only_npoints):
                dic2return["avg_densities"].append(np.mean(np.array(get_point_cloud_density(actual_pointcloud[:,0:3], radius=radius))))
    if(idx_core==-1):
        return dic2return
    else:
        with open("report_core_%i.json" %(idx_core), 'w') as outfile:
            json.dump(dic2return, outfile)
        print("-> Core-%i has finish" %(idx_core))
         
def get_batches(lst, cores):
    """
    Split the list in different batches to fit the defined number of cores 

    :INPUT:
        lst: list, list with the path to the files 
        cores: int, integers grater or iqual to 1
    :OUT:
        list of lists  [[...],[...],[...]]
    """
    batches = cores # int(floor(len(lst)/float(cores)))
    batch_sz = int(len(lst)/batches) if (len(lst)>batches) else 1
    lck = True
    idx_cntr = 0
    lst_batches = []
    tmp = []
    print("-> Batches to generate: %i" %(batches))
    print("  -> Batch size: %i" %(batch_sz))
    while(lck):
        if(len(tmp)<batch_sz):
            tmp.append(lst[idx_cntr])
            idx_cntr =  idx_cntr + 1
        else:
            if(len(lst_batches)<batches):
                lst_batches.append(tmp)
                tmp = []
  
        if(len(lst_batches)==batches):
            lck = False
            if(idx_cntr<len(lst)):
                for a_file in lst[idx_cntr:]:
                    lst_batches[-1].append(a_file)
                    idx_cntr =  idx_cntr + 1
    print("  -> Final conf: B-%i | Bsz[First&Last]:%i&%i | total: %i" %(len(lst_batches), len(lst_batches[0]), len(lst_batches[-1]), idx_cntr))
    return lst_batches

def merge_dictionaries():
    actual_working_path = os.path.join(Path(os.path.dirname(__file__)).resolve(),"test_report_merge_j02rw0_sres008")
    list_of_files = [all_elements for all_elements in os.listdir(actual_working_path) if
                     os.path.isfile( os.path.join(actual_working_path, all_elements))]
    selected_files = [a_file for a_file in list_of_files if("report_core_" in  a_file )]
    merged_dict = {}
    percentageMerge = 0
    idx_fake = 0
    for idx, a_report in enumerate(selected_files):
        print(" -> preparing final report[%i/%i]" %(idx, len(selected_files)))
        file2load = os.path.join(actual_working_path, a_report)
        idx_fake = idx_fake + 1
        percentageMerge = (idx_fake/float(len(selected_files)))
        with open(file2load, "r") as report_file:
            dict_report = json.load(report_file)
        if(idx==0):
            print("  -> merged report started [%.2f%%]" %(percentageMerge))
            merged_dict = dict_report
        else:
            print("  -> concatenating [%.2f%%]" %(percentageMerge))
            for a_key in merged_dict.keys():
                if(a_key in dict_report.keys()):
                    merged_dict[a_key] = merged_dict[a_key] + dict_report[a_key]
                else:
                    merged_dict[a_key] = dict_report[a_key]
    return merged_dict

def main():
    parser = argparse.ArgumentParser("Verify the numper of points of the dataset")
    parser.add_argument("path2pointclouds", type=str, help=" ")
    parser.add_argument("path2out", type=str, help="Path to the output")
    parser.add_argument("--outname", type=str, help="Name for the output file, default:report", default="report")
    parser.add_argument("--hist_name", type=str, help="Histogram Name, default: experiment", default="experiment")
    parser.add_argument("--history_name", type=str, help="History file name, default:history", default="history")
    parser.add_argument("--format", type=str, help="Point clouds format, default:txt", default="txt")
    parser.add_argument("--annColumn", type=int, help="Column of the labeled points, -1 mean there is no label column, default:-1", default=-1)
    parser.add_argument("--cores", type=int, help="Number of cores to use for the task, default:1", default=1)
    parser.add_argument("--radius", type=float, help="Radius of the sphere used to evaluate the point density , unit in meters, default:0.1", default=0.1)
    parser.add_argument("--only_npoints", help="Get only the number of points", action="store_true")
    args = parser.parse_args()
    start_time = datetime.now()
    # Verify that the paths exist 
    stat_folders = verify_paths(args)
    if(stat_folders==-1):
        return 0
    else:
        print("-> Defined folders: OK")
    lst_files = glob.glob(os.path.join(args.path2pointclouds, "*.%s" %(args.format)))
    if(len(lst_files)>0):
        print("-> Found point clouds: %s" %(len(lst_files)))
    else:
        print("[WARNING]-> Any file was found in the selected path: %s" %(args.path2pointclouds))
    if(args.cores == 1):
        dic2save = get_pointcloud_general_characteristics(lst_files, args.annColumn, args.radius, -1, args.only_npoints)
        path2save = os.path.join(args.path2out, "%s.json" %(args.outname))
        with open(path2save, 'w') as outfile:
            json.dump(dic2save, outfile)
    else:
        # Get batches, 
        batches = get_batches(lst_files, args.cores)
        p_list = []
        # fit threats 
        for idx_core in range(args.cores):
            p = Process(target=get_pointcloud_general_characteristics, args=(batches[idx_core], args.annColumn, args.radius, idx_core, args.only_npoints))
            p_list.append(p)
        for a_proc in p_list:
            a_proc.start()
        for a_proc in p_list:
            a_proc.join()
        # Merge dic
        m_dict = merge_dictionaries()
        path2save = os.path.join(args.path2out, "%s.json" %(args.outname))
        with open(path2save, 'w') as outfile:
            json.dump(m_dict, outfile)
    end_time = datetime.now()
    print('Execution time: {}'.format(end_time - start_time))
    print("-> EXIT")
    return 0 

if(__name__=="__main__"):
    dic2test = merge_dictionaries()
    with open("/media/juan/LaCie/results_densities/report.json", 'w') as outfile:
        json.dump(dic2test, outfile)
    #sys.exit(main())