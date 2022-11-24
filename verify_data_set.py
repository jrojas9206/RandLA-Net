import os 
import sys
import glob 
import argparse
import numpy as np 
from math import floor
from pathlib import Path
from multiprocessing import Process, Manager
from sklearn.neighbors import KDTree

def verify_paths(args):
    if(not os.path.isdir(Path(args.path2pointclouds).resolve())):
        print(" ->[ERROR] Input path was not found")
        return -1
    if(not os.path.isdir(args.path2out)):
        print(" ->[WARNING] Output folder was not found, it is going to be created")
        os.mkdir(Path(os.path.path2out).resolve())
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
        float32
    """
    # Get the KDtree representation of the point cloud 
    # and from it begin to estimate the densities 
    # base on the number of points inside the 
    # evaluate sphere 

    kdt_rep = KDTree(pc, leaf_size=2)
    nPts = kdt_rep.query_radius(pc, r=radius, count_only=True)
    lstDens = nPts/( radius**2 )
    return lstDens

def get_pointcloud_general_characteristics(lst_files, out, annColumn=3, radius=0.1, idx_core=-1):
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
            dic2return["avg_densities"] = [np.mean(np.array(get_point_cloud_density(actual_pointcloud, radius=radius)))]
        else:
            dic2return["names"].append(os.path.split(a_file)[-1].split(".")[0])
            dic2return["number_of_points"].append(actual_pointcloud.shape[0])
            dic2return["avg_densities"].append(np.mean(np.array(get_point_cloud_density(actual_pointcloud, radius=radius))))
    out = dic2return
    return dic2return
         
def get_batches(lst, cores):
    batches = int(floor(len(lst)/float(cores)))
    batch_sz = int(len(lst)/batches)
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

def main():
    parser = argparse.ArgumentParser("Verify the numper of points of the dataset")
    parser.add_argument("path2pointclouds", type=str, help=" ")
    parser.add_argument("path2out", type=str, help="Path to the output")
    parser.add_argument("--hist_name", type=str, help="Histogram Name, default: experiment", default="experiment")
    parser.add_argument("--history_name", type=str, help="History file name, default:history", default="history")
    parser.add_argument("--format", type=str, help="Point clouds format, default:txt", default="txt")
    parser.add_argument("--annColumn", type=int, help="Column of the labeled points, -1 mean there is no label column, default:-1", default=-1)
    parser.add_argument("--cores", type=int, help="Number of cores to use for the task, default:1", default=1)
    parser.add_argument("--radius", type=float, help="Radius of the sphere used to evaluate the point density , unit in meters, default:0.1", default=0.1)
    args = parser.parse_args()
    # Verify that the paths exist 
    stat_folders = verify_paths(args)
    if(stat_folders==-1):
        return 0
    else:
        print("-> Defined folders: OK")
    lst_files = glob.glob(os.path.join(args.path2pointclouds, "*.%s" %(args.format)))[:10]
    if(len(lst_files)>0):
        print("-> Found point clouds: %s" %(len(lst_files)))
    else:
        print("[WARNING]-> Any file was found in the selected path: %s" %(args.path2pointclouds))
    if(args.cores == 1):
        dic2save = get_pointcloud_general_characteristics(lst_files, args.annColumn, args.radius, 0)
    else:
        # Get batches, 
        batches = get_batches(lst_files, args.cores)
        a_manager = []
        p_list = []
        # fit threats 
        for idx_core in range(args.cores):
            a_manager.append(Manager().dict())
            p = Process(target=get_pointcloud_general_characteristics, args=(batches[idx_core], a_manager[idx_core], args.annColumn, args.radius, idx_core))
            p_list.append(p)
        for a_proc in p_list:
            a_proc.start()
        for a_proc in p_list:
            a_proc.join()
        # Merge results
        for a_dic in a_manager:
            print(a_dic)
        

    return 0 

if(__name__=="__main__"):
    sys.exit(main())