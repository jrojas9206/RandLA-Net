import numpy
import os
import glob
import helper_ply

def convert(filename, output_dir):

    data = numpy.loadtxt(filename)

    output_filename = os.path.join(output_dir, os.path.basename(filename))

    points = data[:, 0:3]
    colors = data[:, 3:6]


    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    
    helper_ply.write_ply(output_filename, [points, colors], field_names)


if __name__ == "__main__":

    input_dir = "/home/artzet_s/code/dataset/5-fold_labeled/fold_1/"
    output_dir = "/home/artzet_s/code/dataset/5-fold_labeled/fold_1_ply"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filenames = glob.glob(input_dir + "*.txt")
    for filename in filenames:
        print(filename)
        convert(filename, output_dir)
        