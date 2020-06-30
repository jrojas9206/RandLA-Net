import os
import numpy
import glob

if __name__ == "__main__":

    input_dir = "/gpfswork/rech/wwk/uqr22pt/data_field/test/"
    label_dir = "/gpfswork/rech/wwk/uqr22pt/model_RandLA-Net/test/Log_2020-06-18_12-53-43/predictions/"
    output_dir = "/gpfswork/rech/wwk/uqr22pt/pred_RandLA-Net_field_HiRes"


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