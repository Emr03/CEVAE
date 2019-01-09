import tensorflow as tf
from CEVAE.config import FLAGS
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions
import functools
from CEVAE.evaluation import Evaluator

class CEVAE:

    def __init__(self, params):

        self.params = params
        self.latent_size = self.params["latent_size"]
        self.h_size = self.params["hidden_size"]
        self.x_size = self.params["x_size"]
        self.y_size = self.params["y_size"]

        self.x_pl = tf.placeholder(shape=(params["batch_size"], params['x_size']), dtype=tf.float32)
        self.y_pl = tf.placeholder(shape=(params["batch_size"], params['y_size']), dtype=tf.float32)
        self.t_pl = tf.placeholder(shape=(params["batch_size"], 1), dtype=tf.float32)

        self.y_cf_pl = tf.placeholder(shape=(params["batch_size"], params['y_size']), dtype=tf.float32)
        self.mu0_pl = tf.placeholder(shape=(params["batch_size"], params['y_size']), dtype=tf.float32)
        self.mu1_pl = tf.placeholder(shape=(params["batch_size"], params['y_size']), dtype=tf.float32)

        self.elbo = None
        self.pred_log_prob = None
        self.loss = None
        self.train_op = None
        self.build_training_graph()
        self.build_eval_graph()

        self.evaluator = Evaluator(y=self.y_pl, t=self.t_pl, y_cf=self.y_cf_pl, mu0=self.mu0_pl, mu1=self.mu1_pl)

    def make_inference_networks(self, h, x_size, y_size):

        with tf.variable_scope("encoder", reuse=True):

            # initialize encoder_net
            #initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay, scope=None)

            dense = functools.partial(tf.keras.layers.Dense,units=h,
                                      kernel_regularizer=regularizer,
                                      bias_regularizer=regularizer,
                                      activation=tf.nn.elu)

            g1 = tf.keras.Sequential([
                dense(input_shape=(x_size + y_size,)),
                dense(),
                dense(units=FLAGS.latent_size)
            ], name='g1')

            g2 = tf.keras.Sequential([
                dense(input_shape=(FLAGS.latent_size,)),
                dense(),
                dense(units=FLAGS.latent_size),
            ], name='g2')

            g3 = tf.keras.Sequential([
                dense(input_shape=(FLAGS.latent_size,)),
                dense(),
                dense(units=FLAGS.latent_size),
            ], name='g3')

        def encoder(x, t, y):

            shared_rep = g1(tf.concat([x, y], axis=-1))
            latent_code_t1 = g2(shared_rep)
            latent_code_t0 = g3(shared_rep)

            params = (1 - t)*latent_code_t0 + t*latent_code_t1
            approx_posterior = tfd.Bernoulli(logits=params, dtype=tf.float32)
            return approx_posterior

        return encoder

    def make_decoder_networks(self, x_size, y_size, h):

        with tf.variable_scope("decoder", reuse=True):

            # initialize encoder_net
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay, scope=None)

            dense = functools.partial(tf.keras.layers.Dense, units=h,
                                      kernel_regularizer=regularizer,
                                      bias_regularizer=regularizer,
                                      activation=tf.nn.elu)

            proxy_gen_net = tf.keras.Sequential([
                dense(input_shape=(FLAGS.latent_size,)),
                dense(),
                dense(units=2*x_size),
            ], name='proxy_gen_net')

            outcome_gen_net_t1 = tf.keras.Sequential([
                dense(input_shape=(FLAGS.latent_size,)),
                dense(),
                dense(units=2*y_size)
            ], name='outcome_gen_net_t1')

            outcome_gen_net_t0 = tf.keras.Sequential([
                dense(input_shape=(FLAGS.latent_size,)),
                dense(),
                dense(units=2*y_size)
            ], name="outcome_gen_net_t0")

            treatment_gen_net = dense(units=1, activation=tf.nn.sigmoid, name="treatment_gen_net")
            EPS = 1e-03

        def decoder(z):
            """

            :param z: sampled from posterior or prior
            :return: posterior predictive distributions
            """
            print_ops = []

            # p(x|z)
            def x_decoder():
                x_dist_params = proxy_gen_net(z)
                print_ops.append(tf.print(x_dist_params, name="x_dist_params"))
                x_dist = tfd.MultivariateNormalDiag(loc=x_dist_params[..., 0:x_size],
                                              scale_diag=x_dist_params[..., x_size:] + EPS,
                                              name='x_post')

                return x_dist


            # p(t|z)
            def t_decoder():
                treatment_dist_params = treatment_gen_net(z)
                print_ops.append(tf.print(treatment_dist_params, name="t_dist_params"))
                t_dist = tfd.Bernoulli(logits=treatment_dist_params, dtype=tf.float32, name='t_post')
                return t_dist

            # p(y| t, z)
            def y_decoder(t):
                y_dist_params = (1-t)*outcome_gen_net_t0(z) + t*outcome_gen_net_t1(z)
                print_ops.append(tf.print(y_dist_params, name="y_dist_params"))
                y_dist = tfd.MultivariateNormalDiag(loc=y_dist_params[..., 0:y_size],
                                              scale_diag=y_dist_params[..., y_size:] + EPS,
                                              name='y_post')
                return y_dist

            return x_decoder, t_decoder, y_decoder, print_ops

        return decoder

    def make_prediction_networks(self, x_size, y_size, h):

        with tf.variable_scope("predictor", reuse=True):
            # initialize encoder_net
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay, scope=None)

            dense = functools.partial(tf.keras.layers.Dense, units=h,
                                          kernel_regularizer=regularizer,
                                          bias_regularizer=regularizer,
                                          activation=tf.nn.elu)

            g4 = dense(input_shape=(x_size,), units=1, activation=tf.nn.sigmoid, name='g4')

            g5 = tf.keras.Sequential([
                dense(input_shape=(x_size,)),
                dense(),
                dense(units=FLAGS.latent_size)
            ], name='g5')

            g6 = tf.keras.Sequential([
                dense(input_shape=(FLAGS.latent_size,)),
                dense(),
                dense(units=2*y_size)
            ], name='g6')

            g7 = tf.keras.Sequential([
                dense(input_shape=(FLAGS.latent_size,)),
                dense(),
                dense(units=2*y_size)
            ], name='g7')

        def treatment_pred(x):
            with tf.variable_scope("treatment_predictor"):
                treatment_dist_param = g4(x)
                treatment = tfd.Bernoulli(logits=treatment_dist_param, dtype=tf.float32, name="t_pred")
                return treatment

        def outcome_pred(t, x):
            with tf.variable_scope("outcome_predictor"):
                shared_rep = g5(x)
                outcome_dist_params = t*g6(shared_rep) + (1-t)*g7(shared_rep)
                outcome = tfd.MultivariateNormalDiag(loc=outcome_dist_params[..., 0:y_size],
                                                    scale_diag=outcome_dist_params[..., y_size:],
                                                    name='y_pred')

                return outcome

        return treatment_pred, outcome_pred

    def latent_prior(self, latent_size):

        return tfd.Bernoulli(probs=0.5*np.ones(latent_size, dtype=np.float32), dtype=tf.float32)

    def build_training_graph(self):
        """
        train posterior inference (encoder), decoder and prediction networks using training tuples (x, t, y)
        use observed value of t to decode or predict y (teacher forcing)
        :return:
        """

        prior = self.latent_prior(self.latent_size)
        encoder = self.make_inference_networks(x_size=self.x_size, y_size=self.y_size, h=self.h_size)
        decoder = self.make_decoder_networks(x_size=self.x_size, y_size=self.y_size, h=self.h_size)

        # predict t and y just from x
        t_predictor, y_predictor = self.make_prediction_networks(x_size=self.x_size, y_size=self.y_size, h=self.h_size)

        qz = encoder(x=self.x_pl, y=self.y_pl, t=self.t_pl)

        # within-sample posterior predictive distributions, use observed treatment
        x_dec, t_dec, y_dec, print_ops = decoder(z=qz.sample())
        x_post = x_dec()
        t_post = t_dec()
        y_post = y_dec(t=self.t_pl)

        # observational y prediction is not the same as interventional y prediction
        treatment_pred, outcome_pred = t_predictor(x=self.x_pl), y_predictor(t=self.t_pl, x=self.x_pl)

        # for debugging
        self.log_prob_dict = {"x_dist": x_post.log_prob(self.x_pl),
                         "t_dist": t_post.log_prob(self.t_pl),
                         "y_dist": y_post.log_prob(self.y_pl),
                         "kl_div": tfd.kl_divergence(qz, prior)}

        self.elbo = tf.squeeze(x_post.log_prob(self.x_pl)) + \
                    tf.squeeze(t_post.log_prob(self.t_pl)) + \
                    tf.squeeze(y_post.log_prob(self.y_pl)) - \
                    tf.squeeze(tfd.kl_divergence(qz, prior))

        self.pred_log_prob = tf.squeeze(treatment_pred.log_prob(self.t_pl)) + \
                             tf.squeeze(outcome_pred.log_prob(self.y_pl))

        self.loss = -self.elbo - self.pred_log_prob

        self.loss = tf.reduce_mean(self.loss)
        self.elbo = tf.reduce_mean(self.elbo)
        self.pred_log_prob = tf.reduce_mean(self.pred_log_prob)

        tf.summary.scalar("elbo", self.elbo)
        tf.summary.scalar("loss", self.loss)

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.params["learning_rate"], global_step,
                                                   self.params["max_steps"], decay_rate=0.99)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        self.summaries = tf.summary.merge_all()

    def build_eval_graph(self):

        """
        evaluate performance assuming we can only observe x
        :return:
        """
        prior = self.latent_prior(self.latent_size)
        encoder = self.make_inference_networks(x_size=self.x_size, y_size=self.y_size, h=self.h_size)
        decoder = self.make_decoder_networks(x_size=self.x_size, y_size=self.y_size, h=self.h_size)

        # predict t and y just from x
        t_predictor, y_predictor = self.make_prediction_networks(x_size=self.x_size, y_size=self.y_size, h=self.h_size)
        treatment_pred = t_predictor(x=self.x_pl)

        # we have to sample alternative treatments and outcomes to get q(z|x)
        sampled_t = treatment_pred.sample()
        outcome_pred = y_predictor(t=sampled_t, x=self.x_pl)

        # use the encoder to estimate the latent variables
        qz = encoder(x=self.x_pl, t=sampled_t, y=outcome_pred.sample())

        # samples from the posterior for each data entry
        # shape = (n_samples, batch_size, latent_size)
        z_samples = qz.sample()
        print("z_samples shape", z_samples.shape)

        # use the decoder to get posterior distribution for y
        x_dec, t_dec, y_dec, print_ops = decoder(z=z_samples)
        y_post = y_dec(t=self.t_pl)
        self.y_post_mean = y_post.mean()

    def evaluate(self, val_input_fn, L):

        get_val_batch = val_input_fn()

        # interventional prediction
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            val_batch = sess.run(get_val_batch)
            y_pred0 = sess.run(self.y_post_mean, feed_dict={self.t_pl: np.zeros(self.t_pl.shape),
                                                            self.x_pl: val_batch['x']})
            y_pred1 = sess.run(self.y_post_mean, feed_dict={self.t_pl: np.ones(self.t_pl.shape),
                                                            self.x_pl: val_batch['x']})

            # evaluation
            ite, ate, pehe = self.evaluator.calc_stats(ypred1=y_pred1, ypred0=y_pred0)
            print("ite ", ite)
            print("ate ", ate)
            print("pehe", pehe)


    def train(self, epochs, train_input_fn, val_input_fn):

        get_train_batch = train_input_fn()
        get_val_batch = val_input_fn()

        with tf.Session() as sess:

            train_summary_writer = tf.summary.FileWriter('logs/train/', sess.graph)
            val_summary_writer = tf.summary.FileWriter('logs/val/')

            sess.run(tf.global_variables_initializer())
            for i in range(epochs):

                train_batch = sess.run(get_train_batch)

                loss, elbo, log_pred, _ , summary = sess.run([self.loss, self.elbo, self.pred_log_prob,
                                                                      self.train_op, self.summaries],
                                                           feed_dict={self.x_pl: train_batch['x'],
                                                                      self.y_pl: train_batch['y'],
                                                                      self.t_pl: train_batch['t']})

                # print("x_dist: ", x_dist)
                # print("t_dist: ", t_dist)
                # print("y_dist: ", y_dist)
                # print("kl_div: ", kl_div)

                print("train loss", loss)
                print("train elbo", elbo)
                print("train log_pred", log_pred)
                train_summary_writer.add_summary(summary, i)

                #test_eval_summary = sess.run()












