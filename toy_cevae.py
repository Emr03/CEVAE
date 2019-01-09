import tensorflow as tf
from CEVAE.config import FLAGS
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
from CEVAE.toy_cevae_class import *

def build_input_fns(train_data_dir, val_data_dir, batch_size):

    train_data_arr = np.load(train_data_dir)
    train_size = int(train_data_arr['z'].shape[0])

    # preprocess the outcome
    y = train_data_arr['y']
    y_mean, y_std = np.mean(y), np.std(y)
    print(y_mean, y_std)
    y = (y - y_mean) / y_std
    print(y.shape)

    train_data_dict = {'x': train_data_arr['x'],
                       'y': y,
                       'z': train_data_arr['z'],
                       't': train_data_arr['t'],
                       'y_cf': train_data_arr['y_cf'],
                       'mu0': train_data_arr['mu0'],
                       'mu1': train_data_arr['mu1']}

    val_data_arr = np.load(val_data_dir)
    val_size = int(val_data_arr['z'].shape[0])
    y = val_data_arr['y']
    y = (y - y_mean) / y_std
    val_data_dict = {'x': val_data_arr['x'],
                     'y': y,
                     'z': val_data_arr['z'],
                     't': val_data_arr['t'],
                     'y_cf': val_data_arr['y_cf'],
                     'mu0': val_data_arr['mu0'],
                     'mu1': val_data_arr['mu1']}

    print("train_size", train_size)
    print("val_size", val_size)

    def train_input_fn():

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data_dict)\
            .shuffle(train_size).repeat().batch(batch_size)
        iterator = train_dataset.make_one_shot_iterator()
        return iterator.get_next()

    def eval_input_fn():

        val_dataset = tf.data.Dataset.from_tensor_slices(val_data_dict)\
            .shuffle(val_size).repeat().batch(batch_size)
        iterator = val_dataset.make_one_shot_iterator()
        return iterator.get_next()

    return train_input_fn, eval_input_fn

def main(argv):
    params = FLAGS.flag_values_dict()

    data_dir = params["dataset_dir"] + "toy_data/"
    params["x_size"] = 1
    params["y_size"] = 1
    params["latent_size"] = 1

    train_data_dir = data_dir+"train.npz"
    val_data_dir = data_dir+"val.npz"

    train_input_fn, val_input_fn = build_input_fns(train_data_dir=train_data_dir,
                                                   val_data_dir=val_data_dir,
                                                   batch_size=params["batch_size"])

    cevae = CEVAE(params)
    #cevae.train(epochs=10000, train_input_fn=train_input_fn, val_input_fn=val_input_fn)
    cevae.evaluate(val_input_fn, L=1)

if __name__=="__main__":
    tf.app.run(main)