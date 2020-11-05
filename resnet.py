import tensorflow as tf


def residual_unit(inputs, depth, block_num, unit_num):
    f1, f2, f3 = depth
    main_path = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-%s-1' % (block_num, unit_num))(inputs)
    main_path = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-%s-1' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), kernel_initializer='he_uniform',
                                       name='Conv2D_Unit_%s-%s-1' % (block_num, unit_num),kernel_regularizer =
                                       tf.keras.regularizers.l2(l=0.003))(main_path)

    main_path = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-%s-2' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-%s-2' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Conv2D(filters=f2, kernel_size=(3, 3), kernel_initializer='he_uniform',
                                       padding='same', name='Conv2D_Unit_%s-%s-2' % (block_num, unit_num),
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.003))(main_path)

    main_path = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-%s-3' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-%s-3' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Conv2D(filters=f3, kernel_size=(1, 1), kernel_initializer='he_uniform',
                                       name='Conv2D_Unit_%s-%s-3' % (block_num, unit_num),kernel_regularizer
                                       =tf.keras.regularizers.l2(l=0.003))(main_path)

    block_output = tf.keras.layers.Add()([inputs, main_path])

    return block_output

def residual_unit2(inputs, depth, block_num, unit_num):
    f1, f2 = depth
    main_path = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-%s-1' % (block_num, unit_num))(inputs)
    main_path = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-%s-1' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), kernel_initializer='he_uniform',
                                       padding = 'same', name='Conv2D_Unit_%s-%s-1' % (block_num, unit_num),
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.001))(main_path)

    main_path = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-%s-2' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-%s-2' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Conv2D(filters=f2, kernel_size=(3, 3), kernel_initializer='he_uniform',
                                       padding='same', name='Conv2D_Unit_%s-%s-2' % (block_num, unit_num),
                                       kernel_regularizer = tf.keras.regularizers.l2(l=0.001))(main_path)


    block_output = tf.keras.layers.Add()([inputs, main_path])

    return block_output

def residual_unit3(inputs, depth, block_num, unit_num):
    f1, f2 = depth


    main_path = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), kernel_initializer='he_uniform',
                                       padding = 'same', name='Conv2D_Unit_%s-%s-1' % (block_num, unit_num))(inputs)
    main_path = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-%s-1' % (block_num, unit_num))(main_path)
    main_path = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-%s-1' % (block_num, unit_num))(main_path)

    main_path = tf.keras.layers.Conv2D(filters=f2, kernel_size=(3, 3), kernel_initializer='he_uniform',
                                       padding='same', name='Conv2D_Unit_%s-%s-2' % (block_num, unit_num))(main_path)

    main_path = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-%s-2' % (block_num, unit_num))(main_path)



    block_output = tf.keras.layers.Add()([inputs, main_path])
    block_output = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-%s-3' % (block_num, unit_num))(block_output)
    return block_output

def conv_block(inputs, depth, block_num):

    inputs = tf.keras.layers.Conv2D(filters=depth, kernel_size=(3, 3), kernel_initializer='he_uniform',
                                    padding='same', name='Conv2D_Unit_%s' % block_num,
                                    kernel_regularizer = tf.keras.regularizers.l2(l=0.003))(inputs)
    inputs = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s' % block_num)(inputs)
    inputs = tf.keras.layers.Activation('relu', name='Activation_Unit_%s' % block_num)(inputs)
    return inputs


def connection_block(inputs, depth_out, block_num):

    inputs = tf.keras.layers.Conv2D(filters=depth_out, kernel_size=(1, 1), kernel_initializer='he_uniform',
                                    name='Conv2D_Unit_%s-0' % block_num,kernel_regularizer
                                    = tf.keras.regularizers.l2(l=0.003))(inputs)
    inputs = tf.keras.layers.BatchNormalization(axis=-1, name='BN_Unit_%s-0' % block_num)(inputs)
    inputs = tf.keras.layers.Activation('relu', name='Activation_Unit_%s-0' % block_num)(inputs)
    return inputs


def residual_block(inputs, depth, int_unit_num, block_num):

    inputs = connection_block(inputs, depth[2], block_num)
    for i in range(int_unit_num):
        inputs = residual_unit(inputs, depth, block_num, str(i+1))
    return inputs

def residual_block2(inputs, depth, int_unit_num, block_num):

    inputs = connection_block(inputs, depth[1], block_num)
    for i in range(int_unit_num):
        inputs = residual_unit2(inputs, depth, block_num, str(i+1))
    return inputs

def residual_block3(inputs, depth, int_unit_num, block_num):

    inputs = connection_block(inputs, depth[1], block_num)
    for i in range(int_unit_num):
        inputs = residual_unit3(inputs, depth, block_num, str(i+1))
    return inputs


def output_block(inputs):
    inputs = tf.keras.layers.Flatten()(inputs)
    return inputs