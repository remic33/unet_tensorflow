"""Main file to execute model in different modes"""

import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from keras.callbacks import ModelCheckpoint
from skimage import io

from src.data import save_result, test_generator, train_generator
from src.model import Unet


# launch options
flags.DEFINE_enum(
    "mode",
    "train",
    ["train", "predict", "full", "transfer"],
    "Choose training/predict mode",
)
flags.DEFINE_enum(
    "block",
    "resnet",
    ["standard_no_bn", "standard", "resnet", "inception"],
    "Choose type of block",
)
flags.DEFINE_string("path", "data/model/unet_membrane.hdf5", "model path")
flags.DEFINE_enum("padding", "same", ["same", "valid"], "Choose training/predict mode")
flags.DEFINE_integer("lbp", 2, "Number of layers between each pool")
flags.DEFINE_integer("pools", 4, "Number of poolling operations")
flags.DEFINE_integer("epochs", 1, "Number of training epochs")


def main(argv):  # pylint: disable=W0613
    """Main training & predict function. Read readme.md for full usage"""
    logging.info(f"{FLAGS.mode} mode")
    np.random.seed(0)  # random seed for consistency in tests methods
    if FLAGS.mode in ["train", "full"]:
        data_gen_args = dict(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        train_gen = train_generator(
            2, "data/membrane/train", "image", "label", data_gen_args, save_to_dir=None
        )

        unet = Unet(
            layers_between_pool=FLAGS.lbp,
            pool_num=FLAGS.pools,
            block=FLAGS.block,
            padding=FLAGS.padding,
        )
        model = unet.get_model()

    if FLAGS.mode == "transfer":
        unet = Unet(pretrained_weights=FLAGS.path)
        model = unet.get_model()

    if FLAGS.mode in ["train", "full", "transfer"]:
        logging.info(f"Training saves at {FLAGS.path}")
        model_checkpoint = ModelCheckpoint(
            FLAGS.path, monitor="loss", verbose=1, save_best_only=True
        )
        model.summary()
        model.fit(
            train_gen,
            steps_per_epoch=300,
            epochs=FLAGS.epochs,
            callbacks=[model_checkpoint],
        )

    if FLAGS.mode == "predict":
        # init model with pretrained weights
        logging.info(f"Loading pretrained model at {FLAGS.path}")
        model = Unet(pretrained_weights=FLAGS.path).get_model()

    if FLAGS.mode in ["predict", "full"]:
        test_gen = test_generator("data/membrane/test")
        output_size = io.imread("data/membrane/test/0.png").shape
        results = model.predict_generator(test_gen, 30, verbose=1)
        save_result("data/membrane/test", results, output_size=output_size)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
