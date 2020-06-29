import os
import numpy
import glob

if __name__ == "__main__":

    # input_dir = "/home/artzet_s/code/dataset/field_afef_apple_tree_filtered/"
    # label_dir = "/home/artzet_s/code/dataset/labels/Log_2020-06-12_06-25-29/predictions/"
    # output_dir = "/home/artzet_s/code/dataset/randlanet_prediction_2"

    input_dir = "/gpfswork/rech/wwk/uqr22pt/field_afef_apple_tree_filtered/"
    label_dir = "/gpfswork/rech/wwk/uqr22pt/RandLaNet_results/test/Log_2020-06-18_08-11-09/predictions/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/randlanet_prediction_3"


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