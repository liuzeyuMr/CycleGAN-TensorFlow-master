import tensorflow as tf
import ops
import utils

class Generator:
  #生成网络
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size=256):
    self.name = name
    self.reuse = False
    self.ngf = ngf #64
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size #256

  def __call__(self, input): #
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 64)
      d128 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d128')                                 # (?, w/2, h/2, 128)
      d256 = ops.dk(d128, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d256')                                # (?, w/4, h/4, 256)


      if self.image_size <= 128:
        # use 6 residual blocks for 128x128 images
        res_output = ops.n_res_blocks(d256, reuse=self.reuse, n=6)      # (?, w/4, h/4, 128) 残差块
      else:
        # 9 blocks for higher resolution
        res_output = ops.n_res_blocks(d256, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)

      # fractional-strided convolution  decoder
      u128 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 128)
      u64 = ops.uk(u128, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 64)

      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here
      output = ops.c7s1_k(u64, 3, norm=None, #注：这里没有进行归一化
          activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
    # set reuse=True for next call
    self.reuse = True #将重用设置为True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name) #获得判别网络中trainable的变量

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image
