# From https://github.com/PacktPublishing/Machine-Learning-for-Cybersecurity-Cookbook

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dropout, Multiply

def malconv(maxlen=1000,kernel=100,stride=50,filters=128,dense=64):
    import os
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.layers import Dropout, Multiply, SpatialDropout1D
    from livelossplot import PlotLossesKeras

    embedding_dim=8
    vocab_size=256

    inp = Input( shape=(maxlen,))
    emb = Embedding(vocab_size, embedding_dim
                   )( inp )
    filt = Conv1D( filters=filters, kernel_size=kernel, strides=kernel,
                  use_bias=True, activation='relu',
                  padding='same' )(emb)
    attn = Conv1D( filters=filters, kernel_size=kernel, strides=kernel,
                  use_bias=True, activation='sigmoid',
                  padding='same')(emb)
    gated = Multiply()([filt,attn])
    feat = GlobalMaxPooling1D()( gated )
    dense = Dense(dense, activation='relu')(feat)
    outp = Dense(9, activation='softmax')(dense)

    model = Model(inp, outp)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    return model
