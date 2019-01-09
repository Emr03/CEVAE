import tensorflow as tf
from CEVAE.config import flags
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions
from CEVAE.config import FLAGS
from absl import app

def generate(N, filename):

    z = ed.RandomVariable(tfd.Bernoulli(logits=0.5, dtype=tf.float32), sample_shape=(N,1))
    sigma_0 = tf.constant(3.)
    sigma_1 = tf.constant(5.)

    x = ed.RandomVariable(tfd.Normal(loc=z, scale=z*sigma_1 + (1 - z)*sigma_0))
    t = ed.RandomVariable(tfd.Bernoulli(0.75 * z + 0.25 * (1 - z), dtype=tf.float32))

    y = ed.RandomVariable(tfd.Normal(loc = 3 * (z + 2 * (2 * t - 1)),
                                     scale=t*sigma_1 + (1 - t)*sigma_0))

    mu0 = tfd.Normal(loc = 3 * (z - 2), scale = sigma_0).mean()

    mu1 = tfd.Normal(loc = 3 * (z + 2), scale=sigma_1).mean()

    y_cf = ed.RandomVariable(tfd.Normal(loc = 3 * (z + 2 * (2 * (1-t) - 1)),
                                     scale=(1-t)*sigma_1 + t*sigma_0))

    sess = tf.Session()
    with sess.as_default():

        z_val, x_val, t_val, y_val, y_cf_val, mu0_val, mu1_val = sess.run([z.value, x.value, t.value,
                                                                           y.value, y_cf.value,
                                                                           mu0, mu1])
        #print(z_val, x_val, t_val, y_val)

    data_dir = FLAGS.dataset_dir + "toy_data/" + filename
    #print(x_val, y_val, t_val, z_val)
    np.savez(data_dir, z=z_val, x=x_val, t=t_val, y=y_val, y_cf=y_cf_val, mu0=mu0_val, mu1=mu1_val)

def main(argv):
    generate(N=3000, filename="train")
    generate(N=600, filename="val")


if __name__ == "__main__":
   app.run(main)


