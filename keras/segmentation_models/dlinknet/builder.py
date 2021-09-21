from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add

from .blocks import DecoderBlock
from ..utils import get_layer_number, to_tuple


def build_dlinknet(backbone,
                  classes,
                  skip_connection_layers,
                  decoder_filters=(None, None, None, None, 16),
                  upsample_rates=(2, 2, 2, 2, 2),
                  n_upsample_blocks=5,
                  upsample_kernel_size=(3, 3),
                  upsample_layer='upsampling',
                  activation='sigmoid',
                  use_batchnorm=True):

    input = backbone.input
    x = backbone.output

    filters = K.int_shape(x)[-1]
    dlt1 = Conv2D(filters, (3, 3), padding='same', dilation_rate=1, name='dilate_conv1')(x)
    dlt2 = Conv2D(filters, (3, 3), padding='same', dilation_rate=2, name='dilate_conv2')(dlt1)
    dlt4 = Conv2D(filters, (3, 3), padding='same', dilation_rate=4, name='dilate_conv3')(dlt2)
    dlt8 = Conv2D(filters, (3, 3), padding='same', dilation_rate=8, name='dilate_conv4')(dlt4)

    x = Add()([x, dlt1, dlt2, dlt4, dlt8])

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = DecoderBlock(stage=i,
                         filters=decoder_filters[i],
                         kernel_size=upsample_kernel_size,
                         upsample_rate=upsample_rate,
                         use_batchnorm=use_batchnorm,
                         upsample_layer=upsample_layer,
                         skip=skip_connection)(x)

    x = Conv2D(classes, (3, 3), padding='same', name='final_conv')(x)
    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model
