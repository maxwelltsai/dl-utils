import numpy as np 
import h5py 
import os 
import glob 
import tensorflow as tf 
import argparse
import sys 


class JPEGDecoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
    
    @staticmethod
    def imread(filename, decoder):
        # Read the image file.
        with tf.gfile.FastGFile(filename, 'rb') as f:
            image_data = f.read()
        # Decode the RGB JPEG.
        image = decoder.decode_jpeg(image_data)
        return image
        
        

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='dir_name', type=str, help='name of the image dir')
parser.add_argument('-f', dest='output_fn', type=str, default='output.hdf5', help='name of the output file')
args = parser.parse_args()

dir_list = glob.glob(os.path.join(os.path.abspath(args.dir_name), 's_*'))
if len(dir_list) == 0:
    dir_list.append(os.path.abspath(args.dir_name))

n_channels = 3
n_camera_pos = 14
image_dim = 2048

decoder = JPEGDecoder()

with h5py.File(args.output_fn, 'w') as h5f:
    for wd_i, working_dir in enumerate(dir_list):
        dir_name = os.path.basename(working_dir)
        size_ratio = float(dir_name.split('_')[1])
        mass_ratio = float(dir_name.split('_')[3])
        
        h5g = h5f.create_group(dir_name)
        
        for cpos in range(n_camera_pos):
            files_list = glob.glob(os.path.join(working_dir, ('*-*-%02d.jpg' % cpos)))
            n_images = len(files_list)
            # allocate arrays
            data_t = np.zeros(n_images, dtype=np.int16)
            data = np.zeros((n_images, image_dim, image_dim, n_channels), dtype=np.uint8)
            
            for i, fullpath in enumerate(files_list):
                if not os.path.isfile(fullpath):
                    continue
                fn = os.path.basename(fullpath)
                t_myr = int(fn.split('.')[0].split('-')[1])
                
                # read the image data from file 
                img = JPEGDecoder.imread(fullpath, decoder)
                print('Processing ensemble %s [%d/%d], file %s [%d/%d], camera_pos = %02d [%d/%d]' % (dir_name, wd_i, len(dir_list),
                                                                                            fn, i, n_images, 
                                                                                            cpos, cpos, n_camera_pos))
                if n_channels == 1:
                    # convert to single-channel grey scale images
                    data[i, :, :, 0] = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] * 0.113 * img[:, :, 2]
                else:
                    data[i] = img
                data_t[i] = t_myr
            print('Saving compressed datasets...')
            h5g.create_dataset('images_camera_%02d' % cpos, data=data, chunks=True, compression='gzip', compression_opts=3)
            h5g.create_dataset('t_myr_camera_%02d' % cpos, data=data_t)
            print('Done.\n')