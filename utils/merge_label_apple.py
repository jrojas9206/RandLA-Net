import os
import sys
import argparse
import numpy
import glob

def merge_pointCloudAndLabels(input_dir, label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filenames = glob.glob(label_dir + "*.labels")
    print(*filenames, sep="\n")
    for i, filename in enumerate(filenames):
        basename = os.path.basename(filename)[:-7]
        data_filename = os.path.join(input_dir, basename + ".txt")

        npy_filename = data_filename[:-3] + 'npy'
        if os.path.exists(npy_filename):
            data = numpy.load(npy_filename)
        else:
            data = numpy.loadtxt(data_filename)
            numpy.save(npy_filename, data)

        label = numpy.loadtxt(filename)
        print("Number of apple point : ", numpy.count_nonzero(label))
        x = numpy.column_stack([data[:, 0:3], label])

        output_filename = os.path.join(output_dir, basename + '.txt')
        numpy.savetxt(output_filename, x)
        print("{}/{} : {}".format(i, len(filenames), output_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Label Merging - RandLA-NET")
    parser.add_argument("--inputDir_labels", type=str, help="Path to the predicted labels", default="./randlanet/predictions/")
    parser.add_argument("--inputDir_pointc", type=str, help="Path to the point clouds", default="./data/realData/test/")
    parser.add_argument("--outputDir", type=str, help="Path to write the new files", default="./output/")
    args = parser.parse_args()

    merge_pointCloudAndLabels(args.inputDir_pointc, args.inputDir_labels, args.outputDir)

