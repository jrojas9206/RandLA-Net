import os
import sys
import argparse
from launcher_apple_trees import *


def main(argv):
    parser.add_argument('--gpu'         , type=int,     help='GPU ID [default: 0]',         default=0)
    parser.add_argument('--mode'        , type=str,     help='options: train, test, vis',   default='train')
    parser.add_argument('--model_path'  , type=str,     help='pretrained model path',       default='None')
    parser.add_argument('--inputDir'    , type=str,     help="Path to the folder with the train/test/validation & the input folders", default=None)
    parser.add_argument("--outputDir"   , type=str,     help="Path to the output folder",   default="./output/")
    parser.add_argument("--protocol"    , type=str,     help="Measurement protocol 'synthetic'|'field_xyz'|'field'", default="synthetic")
    parser.add_argument("--trainFromCHK", type=bool,    help="Continue the training from a given checkpoint, True/False", default=False)
    parser.add_argument("--verbose"     , type=bool,    help="Print messages of some of the step", default=True   )
    args = parser.parse_args()
    # 
    print("-> RandLA-NET")
    print(" -> GPU[ID]: %i" %args.gpu)
    print(" -> Mode[train/test/vis]: %s" %args.mode)
    print(" -> Output[Path]: %s" %args.outputDir)
    print("  -> Status: %s" %("OK" if os.path.isdir(args.inputDir) else "Is going to be created")) 
    print(" -> Input [Path]: %s" %("None" if args.inputDir is None else args.inputDir))
    print("  -> Status: %s"%("OK" if os.path.isdir(args.inputDir) else "Error"))
    print(" -> Protocol: %s" %args.protocol)
    # improve!!
    param = {"gpu":args.gpu, "mode":args.mode, "model_path":args.model_path, "path2data":args.inputDir, 
             "path2output": args.outputDir, "protocol":args.protocol, "restoreTrain":args.restoreTrain}
    # Lauch the train or the prediction 
    launch_action(param, verbose=args.verbose)
    return 0

if(__name__=="__main__"):
    sys.exit(main(sys.argv))