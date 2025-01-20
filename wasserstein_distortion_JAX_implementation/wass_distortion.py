# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# edit by Yang Qiu for Wasserstein distortion
#
# ==============================================================================

import jax
import jax.numpy as jnp
import h5py
import flax
import flax.linen as nn
import os
import pickle
import h5py

# =================================================
# discrete gauss pmf

# @jax.jit
# def get_size_list(x):
#   print(jnp.shape(x))
#   H,W,Ch = jnp.shape(x)
#   size_list = [[int(H),int(W)]]
#   assert H >= 16
#   assert W >= 16
#   for i in range(4):
#     size_list.append([int(H/(2**(i+1))),int(W/(2**(i+1)))])
#   return size_list
# use this if feature layer sizes unknown

if os.path.exists('pmf.p'):
  pmf_list = pickle.load(open('pmf.p','rb'))
  # plug in if pre-calculated pooling pmf for this size_list existed already

else:

  size_list = [[512,512],[256,256],[128,128],[64,64],[32,32]] # hard-coding list of feature layer sizes for efficiency

  def get_pmf(size_list): # find pooling pmf
    sigma = 8
    pmf_list = {}
    for [H,W] in size_list:
      H_half = int(H/2)
      W_half = int(W/2)
      pmf = jnp.array([[(1/(2*sigma**2))*jnp.exp(-((i_ref_p-0)**2+(j_ref_p-0)**2)/2*sigma**2)\
              for j_ref_p in range(-W_half,W_half)]\
              for i_ref_p in range(-H_half,H_half)])
      pmf = jnp.expand_dims(pmf/jnp.sum(pmf),axis=2)
      pmf = jnp.expand_dims(pmf,axis=0)
      pmf_list[H] = pmf
    return pmf_list

  pmf_list = get_pmf(size_list)
  pickle.dump(pmf_list,open('pmf.p','wb')) # save pooling pmf to file for efficiency

# =================================================
# VGG structure

class VGG(nn.Module):

  def setup(self):
    self.param_dict = h5py.File('vgg19_norm_weights.h5', 'r')
    self.dtype = 'float32'

  @nn.compact
  def __call__(self, x, train=False):

    mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(x.dtype)
    std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(x.dtype)
    x = (x - mean) / std

    act = [x]

    x,act = self._conv_block(x, features=64, num_layers=2, block_num=1, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=128, num_layers=2, block_num=2, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=256, num_layers=4, block_num=3, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=512, num_layers=4, block_num=4, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    x,act = self._conv_block(x, features=512, num_layers=4, block_num=5, act=act, dtype=self.dtype)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    return act

  def _conv_block(self, x, features, num_layers, block_num, act, dtype='float32'):
    for l in range(num_layers):
      layer_name = f'block{block_num}_conv{l + 1}'
      w = lambda *_ : jnp.array(self.param_dict[layer_name][layer_name]['kernel:0']) 
      b = lambda *_ : jnp.array(self.param_dict[layer_name][layer_name]['bias:0']) 
      x = nn.Conv(features=features, kernel_size=(3, 3), kernel_init=w, bias_init=b,\
        padding='same', name=layer_name, dtype=dtype)(x)
      act.append(x)
      x = nn.relu(x)
    return x,act

# =================================================
# wasserstein distortion

def compute_features(image):
  # vgg featues for image
  image = image[None]
  vgg19 = VGG()
  init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
  params = vgg19.init(init_rngs, image)
  features = vgg19.apply(params, image)
  return [jnp.squeeze(f, 0).transpose((2, 0, 1)) for f in features]

def compute_stats(features,pmf_list):
  # moments for all layer outputs
  means = []
  variances = []
  for f in features:
    H = jnp.shape(f)[1]
    pmf = pmf_list[H]
    f = jnp.expand_dims(f,axis=-1)
    squared = jnp.square(f)
    m = jnp.sum(f*pmf,axis=(1,2))
    p = jnp.sum(squared*pmf,axis=(1,2))
    v = p - jnp.square(m)
    means.append(m)
    variances.append(v)
  return means,variances

def wasserstein_distortion(features_a, features_b,pmf_list):
  # wasserstein distortion
  means_a, variances_a = compute_stats(features_a, pmf_list)
  means_b, variances_b = compute_stats(features_b, pmf_list)
  wd_maps = []
  assert len(means_a) == len(means_b) == len(variances_a) == len(variances_b)
  for ma, mb, va, vb in zip(means_a, means_b, variances_a, variances_b):
    sa = jnp.sqrt(va + 1e-4)
    sb = jnp.sqrt(vb + 1e-4)
    wd_maps.append(jnp.square(ma - mb) + jnp.square(sa - sb))
  dist = 0.
  for i, wd_map in enumerate(wd_maps):
    weight = 10**(2 - int(i/5))
    dist += jnp.sum(weight * wd_map)
  return dist

def loss_fn(a,b):
  # loss function 
  features_a = compute_features(a)
  features_b = compute_features(b)
  l = wasserstein_distortion(features_a,features_b,pmf_list)
  return l