from absl import flags

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")

flags.DEFINE_integer(
    "max_steps", default=5001, help="Number of training steps to run.")

flags.DEFINE_integer(
    "latent_size",
    default=1,
    help="Number of dimensions in the latent code (z).")

flags.DEFINE_string(
    "activation",
    default="leaky_relu",
    help="Activation function for all hidden layers.")

flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")

flags.DEFINE_integer(
    "hidden_size",
    default=20,
    help="number of hidden units"
)

flags.DEFINE_integer(
    "n_samples", default=16, help="Number of samples to use in encoding.")

flags.DEFINE_integer(
    "mixture_components",
    default=16,
    help="Number of mixture components to use in the prior. Each component is "
         "a diagonal normal distribution. The parameters of the components are "
         "intialized randomly, and then learned along with the rest of the "
         "parameters. If `analytic_kl` is True, `mixture_components` must be "
         "set to `1`.")

flags.DEFINE_string(
    "model_dir",
    default="logs/",
    help="Directory to save checkpoints and logs"
)

flags.DEFINE_float(
    "weight_decay",
    default=1.0e-3,
    help="Weight decay parameter"
)

flags.DEFINE_string(
    "dataset_dir",
    default="datasets/",
    help="dataset directory"
)

FLAGS = flags.FLAGS
