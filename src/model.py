"""Unet class file"""
from typing import List

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Cropping2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Dropout,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import Tensor
from tensorflow.keras.layers import Resizing


class Unet:  # pylint: disable=C0103
    """
    Unet model class
    """

    def __init__(
        self,
        layers_between_pool: int = 2,
        pool_num: int = 4,
        block: str = "standard",
        pretrained_weights: str = None,
        input_size: tuple = (256, 256, 1),
        padding: str = "same",
    ):
        """
        Init Unet Model
        Args:
            layers_between_pool: number of conv layer between each pooling operation
            pool_num: number of pooling operation
            block: type of block chosen by operator
            pretrained_weights: path to hfd5 trained model
            input_size: input image size in pixel & dimensions
            padding: type of padding
        """
        if layers_between_pool < 1:
            raise ValueError("Layers between pooling number should be a non-zero "
                             "natural integer")
        self.layers_between_pool = layers_between_pool
        self.pool_num = pool_num
        self.padding = padding
        self.input_size = input_size
        self.block = block

        if pretrained_weights is not None:
            self.model = load_model(pretrained_weights)
        elif self.block in ["standard", "standard_no_bn"]:
            self.model = self.standard_model()

        elif self.block == "resnet":
            if layers_between_pool < 1:
                raise ValueError("Layers between pooling number should be a non-zero "
                                 "natural integer")
            if layers_between_pool % 2 != 0:
                raise ValueError(
                    "Layers between pooling number for resnet block should "
                    "be multiple of 2")
            self.model = self.resnet_model()

        elif self.block == "inception":
            self.model = self.inception_model()

    def get_model(self) -> Model:
        """
        Function to get model
        Returns: Model

        """
        return self.model

    @staticmethod
    def resnet_block(
        inputs: Tensor, padding: str, filters: int = 64, conv_input: bool = True
    ) -> Tensor:
        """
        Resnet block input -> conv -> BN -> activation -> conv -> BN -> activation ->
            tmp_output -> Add(input, tmp_output) -> output
        Args:
            inputs: input tensor
            padding: padding type
            filters: filters number
            conv_input: add convolution to input if needed

        Returns: Resnet block tensor

        """
        x = Conv2D(
            filters,
            3,
            activation="relu",
            padding=padding,
            kernel_initializer="he_normal",
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters,
            3,
            activation="relu",
            padding=padding,
            kernel_initializer="he_normal",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        if conv_input:
            inputs = Conv2D(
                filters,
                3,
                activation="relu",
                padding=padding,
                kernel_initializer="he_normal",
            )(inputs)
        Add()([inputs, x])
        return x

    @staticmethod
    def standard_block(
        inputs: Tensor, padding: str, filters: int = 64, bn: bool = True
    ) -> Tensor:
        """
        Standard block of convolution input -> convolution (conv) -> batch normalization
            (BN) (optional) -> activation -> output
        Args:
            inputs: input tensor
            padding: padding type, valid or same
            filters: number of filter
            bn: batch normalization boolean

        Returns: Standard block tensor

        """
        x = Conv2D(
            filters,
            3,
            activation="relu",
            padding=padding,
            kernel_initializer="he_normal",
        )(inputs)
        if bn:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def standard_model(self) -> Model:
        """
        Create Unet standard model with option provided at creation
        Returns: Unet model with standard blocks

        """
        # Init connection with input
        contracting_convs = []
        x = inputs = Input(self.input_size)
        filters = 64  # init filters number
        # Contracting path with standard blocks
        for _ in range(self.pool_num):
            conv = self.multiple_standard(
                inputs=x, layers_between_pool=self.layers_between_pool, filters=filters
            )
            contracting_convs.append(conv)
            x = MaxPooling2D(pool_size=(2, 2))(conv)
            filters = filters * 2
        x = self.multiple_standard(
            inputs=x, layers_between_pool=self.layers_between_pool, filters=filters
        )
        # First Dropout as in original code
        x = Dropout(0.5)(x)
        # Expensive path & output
        model = self.expanding(
            x=x,
            contracting_convs=contracting_convs,
            filters=filters,
            layers_between_pool=self.layers_between_pool,
            padding=self.padding,
            inputs=inputs,
            dropout=True,
        )
        return model

    def resnet_model(self):
        """
        Create Unet resnet model with option provided at creation
        Returns: Unet model with resnet blocks
        """
        # Init connection with input
        contracting_convs = []
        x = inputs = Input(self.input_size)
        filters = 64  # init filters number
        # Contracting path with standard blocks
        conv_input = False
        for _ in range(self.pool_num):
            if _ > 0:
                conv_input = True
            conv = self.multiple_resnet(
                inputs=x,
                layers_between_pool=self.layers_between_pool,
                filters=filters,
                conv_input=conv_input,
            )
            contracting_convs.append(conv)
            x = MaxPooling2D(pool_size=(2, 2))(conv)
            filters = filters * 2

        x = self.multiple_resnet(
            inputs=x, layers_between_pool=self.layers_between_pool, filters=filters
        )
        # Expensive path & output
        model = self.expanding(
            x=x,
            contracting_convs=contracting_convs,
            filters=filters,
            layers_between_pool=self.layers_between_pool,
            padding=self.padding,
            inputs=inputs,
        )
        return model

    def expanding(
        self,
        x: Tensor,
        contracting_convs: List[Tensor],
        filters: int,
        layers_between_pool: int,
        padding: str,
        inputs: Tensor,
        dropout: bool = False,
    ) -> Model:
        """
        Expanding part of unet model with classical conv2d. Number of convolution
            provided at model init
        Args:
            x: input Tensor
            dropout: Boolean for dropout layer. Currently, only use in standard model.
                Can be applied to
                resnet model.
            contracting_convs: preceding convolution to merge with upsamples
            filters: number of filter at the beginning of Expanding part
            layers_between_pool: layers between each upsamble, mimic pools layers
            padding: padding option
            inputs: initial model inputs

        Returns: Model

        """
        contracting_convs.reverse()
        for layer in contracting_convs:
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(
                filters,
                2,
                activation="relu",
                padding=padding,
                kernel_initializer="he_normal",
            )(x)
            if dropout:
                x = Dropout(0.5)(x)
                dropout = False
            # First layer is already created, so range is less 1
            for _ in range(layers_between_pool - 1):
                x = Conv2D(
                    filters,
                    3,
                    activation="relu",
                    padding=padding,
                    kernel_initializer="he_normal",
                )(x)
            # Crop and concatenate with custom cropping (separate crop in case of
            # odd difference between layer)
            if padding == "valid":
                layer = self.crop(layer, x)
            x = concatenate([layer, x], axis=3)
            filters = filters // 2

        # Outputs layers
        if padding == "valid":
            x = Resizing(inputs.shape[1], inputs.shape[2])(x)
        x = Conv2D(1, 1, activation="sigmoid")(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model

    def inception_model(self) -> Model:
        """
        Inception model based. Same expanding than regular unet, the only change is
            blocks that are
            inception inspired.
        Returns: Unet inception model

        """
        # Init connection with input
        contracting_convs = []
        inputs = Input(self.input_size)
        filters = 64  # init filters number
        # Contracting path with standard blocks
        x = Conv2D(
            filters,
            2,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        x = BatchNormalization()(x)
        contracting_convs.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        filters = filters * 2

        for _ in range(self.pool_num - 1):
            x = self.inception_block(inputs=x, filters=filters)
            contracting_convs.append(x)
            filters = filters * 2
            x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(
            filters,
            2,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = BatchNormalization()(x)
        model = self.expanding(
            x,
            contracting_convs,
            filters,
            padding="same",
            inputs=inputs,
            layers_between_pool=self.layers_between_pool,
        )
        return model

    @staticmethod
    def inception_block(inputs: Tensor, filters: int) -> Tensor:
        """
        Inception inspired block. Consist of 4 concurrent group of layers with
            different kerner size that are then concatenate before max pooling
        Args:
            inputs: input Tensor
            filters: number of filter for convolution layers.

        Returns: Tensor with inception block added

        """
        filters = filters // 4
        a = Conv2D(
            filters,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        a = BatchNormalization()(a)
        a = Conv2D(
            filters,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(a)
        a = BatchNormalization()(a)

        b = Conv2D(
            filters,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        b = BatchNormalization()(b)
        b = Conv2D(
            filters,
            5,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(b)
        b = BatchNormalization()(b)

        c = Conv2D(
            filters,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        c = BatchNormalization()(c)

        d = Conv2D(
            filters,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        d = BatchNormalization()(d)
        d = Conv2D(
            filters,
            1,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        d = BatchNormalization()(d)
        x = concatenate([a, b, c, d], axis=3)
        return x

    @staticmethod
    def crop(layer: Tensor, x: Tensor) -> Tensor:
        """
        Crop tensor to fit another tensor to concatenate with
        Args:
            layer: tensor to crop
            x: tensor to fit

        Returns: Cropped layer tensor
        """
        crop_1 = int((layer.shape[1] - x.shape[1]) // 2)
        crop_2 = int(layer.shape[1] - x.shape[1] - crop_1)
        crop_3 = int((layer.shape[2] - x.shape[1]) // 2)
        crop_4 = int(layer.shape[2] - x.shape[1] - crop_3)
        cropping_tuple = ((crop_1, crop_2), (crop_3, crop_4))
        return Cropping2D(cropping=cropping_tuple)(layer)

    def multiple_standard(
        self, inputs: Tensor, layers_between_pool: int, filters: int
    ) -> Tensor:
        """
        Function to create multiple layer
        Args:
            inputs: input tensor
            layers_between_pool: number of convolution layer between each pooling
                operation
            filters: number of filter

        Returns: multiple standard layer Tensor

        """
        if self.block == "standard_no_bn":
            bn = False
        else:
            bn = True

        x = inputs
        for _ in range(layers_between_pool):
            x = self.standard_block(
                inputs=x, padding=self.padding, filters=filters, bn=bn
            )
        return x

    def multiple_resnet(
        self,
        inputs: Tensor,
        layers_between_pool: int,
        filters: int,
        conv_input: bool = True,
    ) -> Tensor:
        """
        Function to create multiple resnet layers
        Args:
            conv_input: Boolean to indicate if adding a convolution is necessary to
                input before add operation
            inputs: input tensor
            layers_between_pool: number of layer between each pool. Should be even
                (resnet block has 2 conv)
            filters: number of filter

        Returns: multiple block tensor

        """

        x = inputs
        for _ in range(layers_between_pool // 2):
            x = self.resnet_block(
                inputs=x, padding=self.padding, filters=filters, conv_input=conv_input
            )
        return x
