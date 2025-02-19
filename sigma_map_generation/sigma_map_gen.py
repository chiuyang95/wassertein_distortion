'''
    @author: Yang Qiu and Ziyuan Lin

    code for 'Wasserstein distortion with Intrinsic sigma-maps'
''' 

import numpy as np 
import tensorflow as tf
import keras.utils as image
from PIL import Image
import math
import sys
import os
import time
import datetime


class SigmaMapGen(object):

    def __init__(self, tex_path):
        '''
            read source image and get properties
            tex_path: path to source image
        '''
        self.tex_path = tex_path
        filename = os.path.basename(tex_path)
        filename = os.path.splitext(filename)[0]
        self.filename = filename


    def preprocess_image(self, img_path):
        '''
            read the image
        '''
        img = image.load_img(img_path)
        img = image.img_to_array(img)
        return img


    def generate_sigma_sequence(self, max_sigma):
        '''
            generate sequence of possible sigma values
        '''
        sigma_sequence = [1.0, 1.5, 2.0, 4.0, 7.0, 10.0]
        increment = 5.0
        increment_steps = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        increment_index = 0
        step_counter = 0
        current_sigma = sigma_sequence[-1] + increment
        while current_sigma <= max_sigma:
            sigma_sequence.append(current_sigma)
            step_counter += 1
            if step_counter % 5 == 0 and increment_index < len(increment_steps) - 1:
                increment_index += 1
                increment = increment_steps[increment_index]
            current_sigma += increment
            if current_sigma >= max_sigma:
                sigma_sequence.append(max_sigma)
                break
        return sigma_sequence


    def get_pmf(self, sigma_sequence):
        '''
            generate pooling pmf
            for each possible sigma values, generate a 2d centered discretized gaussian pmf with variance sigma squared
            of size 2*sigma towards each direction (since this would cover 91% of total probability mass by 3 sigma rule)
        '''
        pmf_lists = {}
        for sigma in sigma_sequence:
            kernel_size = int(4 * sigma - 1)
            current_height = int(kernel_size / 2)
            current_width = int(kernel_size / 2)
            pmf_current = np.array([[np.exp(-((i_ref_p)**2 + (j_ref_p)**2) / (2*sigma**2)) \
                                  for j_ref_p in range(-current_width, current_width + 1)] \
                                 for i_ref_p in range(-current_height, current_height + 1)])
            pmf_current = pmf_current / np.sum(pmf_current)
            pmf_current = np.expand_dims(pmf_current, axis=-1)
            pmf_current = np.expand_dims(pmf_current, axis=-1)
            pmf_current = np.repeat(pmf_current, 3, axis=-1)
            pmf_current_tf = tf.convert_to_tensor(pmf_current, dtype=tf.float32)
            pmf_lists[sigma] = pmf_current_tf
        return pmf_lists


    def get_moments(self,tex,pmf):
        '''
            calculate first two moments via adequately padding the source image and convolving with the pmf
            the first two moments are needed for gaussianized wasserstein distortion
        '''
        tex = tf.expand_dims(tex, axis=0)
        kernel_size = len(pmf)
        padding_size = int((kernel_size + 1) / 2) - 1
        padding = tf.constant([[0, 0], [padding_size, padding_size],\
                        [padding_size, padding_size], [0, 0]])
        tex_padded = tf.pad(tex, padding, "SYMMETRIC")
        mean_tex = tf.nn.conv2d(tex_padded, pmf, strides=[1, 1, 1, 1], padding='VALID')
        mean_tex_padded = tf.pad(mean_tex, padding, "SYMMETRIC")
        squared_diff_tex = (tex_padded - mean_tex_padded) ** 2
        var_tex = tf.maximum(tf.nn.conv2d(squared_diff_tex, pmf,\
                                    strides=[1, 1, 1, 1], padding='VALID'), 0)
        return mean_tex,var_tex


    def get_wass_dist(self, tex, sigma_sequence):
        '''
            get list of wasserstein distortions of all adjacent sigma pairs from sigma_sequence
        '''
        pmf_lists = self.get_pmf(sigma_sequence)
        mean_tex_all = []
        var_tex_all = []
        for sigma in sigma_sequence: # traverse through all sigma values, and calculate moments

            pmf = pmf_lists[sigma]
            mean_tex_tmp,var_tex_tmp = self.get_moments(tex,pmf)
            mean_tex_all.append(mean_tex_tmp)
            var_tex_all.append(var_tex_tmp)

        mean_tex_all = np.concatenate(mean_tex_all,axis=0)
        var_tex_all = np.concatenate(var_tex_all,axis=0)
        wd_sigma_list = (mean_tex_all[1:,:,:,:] - mean_tex_all[:-1,:,:,:])**2 +\
                                var_tex_all[1:,:,:,:] + var_tex_all[:-1,:,:,:] -\
                                2*tf.math.sqrt(var_tex_all[1:,:,:,:]*var_tex_all[:-1,:,:,:])
        wd_sigma_list = tf.math.reduce_mean(wd_sigma_list,axis=3) 
                                            # list of wass. distortions of all sigma pairs
        return wd_sigma_list

    def get_sigma_map(self, tex, threshold):
        '''
            generates sigma map from original picture
        '''
        [height, width] = [tex.shape[0], tex.shape[1]]
        sigma_sequence = self.generate_sigma_sequence(np.min([height,width])/2.) 
                                    # set max sigma value to be half the image height
        sigma_map = np.zeros([height, width])
        wd_sigma_list = self.get_wass_dist(tex, sigma_sequence)

        for x in range(height):
            for y in range(width):
                '''
                    algorithm in paper
                '''
                pixel_wd_values = wd_sigma_list[:,x,y].numpy().tolist()
                wd_values_under_5 = wd_sigma_list[:4,x,y].numpy().tolist()

                max_wd_under_5 = max(wd_values_under_5) if wd_values_under_5 else 0

                # Check if max_wd_under_5 is among the top 5 values in the full list
                top_values = sorted(pixel_wd_values, reverse=True)[:5]
                is_top_five = max_wd_under_5 in top_values

                # Check if max_wd_under_5 is the overall maximum
                overall_max = max(pixel_wd_values)
                is_overall_max = max_wd_under_5 == overall_max

                differences = np.diff(pixel_wd_values)

                # Find the first index where sigma is greater than 10
                if is_top_five:
                    start_index = pixel_wd_values.index(max_wd_under_5) + 1
                    discount = 3/7
                    overall_max = overall_max*discount
                else:
                    start_index = next(
                        (index-1 for index, sigma in enumerate(sigma_sequence) if sigma >= 15),
                        len(sigma_sequence) - 1
                    )

                trend_changes = []

                i = 0
                while i < len(differences):
                    if differences[i] > 0:
                        start = i
                        while i < len(differences) and differences[i] > 0:
                            i += 1
                        trend_changes.append((start, i-1))
                    i += 1

                if is_overall_max:
                    found_significant = False
                    index = pixel_wd_values.index(overall_max) + 1
                    for i in range(index, len(sigma_sequence) - 1):
                        if pixel_wd_values[i] * 10 > max_wd_under_5:
                            sigma_map[x, y] = sigma_sequence[i]
                            found_significant = True
                            break
                    if not found_significant:
                        sigma_map[x, y] = 0
                    continue

                for start, end in trend_changes:
                    if start >= start_index or (start <= start_index <= end):
                        if pixel_wd_values[end] / max_wd_under_5 > threshold:
                            sigma_map[x, y] = sigma_sequence[end + 1]
                            break

        return sigma_map
        

    def buildSigmaMap(self):
        '''
            main function
        '''
        start = time.time()
        tex_img = self.preprocess_image(self.tex_path)
        tex_pixels = tf.squeeze(tex_img)/255.
        threshold = 0.1
        sigma_map = self.get_sigma_map(tex_pixels, threshold)
        np.save(self.filename+'_sigma_map.npy', sigma_map)

        sigma_map_normalized = (sigma_map * 255 / np.max(sigma_map)).astype(np.uint8)
        im = Image.fromarray(sigma_map_normalized, mode='L')
        fname = self.filename + '_sigma_map_heatmap.png'
        im.save(fname)
        print('Image saved as', fname)
        end = time.time()
        print('get a sigma value done with {}'.format(str(datetime.timedelta(seconds=end - start))))


smap = SigmaMapGen(tex_path='./source/'+sys.argv[1]+'.jpg')
smap.buildSigmaMap()