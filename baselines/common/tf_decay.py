
import tensorflow as tf


def schedule(decay='constant', init_lr=1e-4, global_step=None, decay_steps=1e8, end_lr=1e-8):
    """
    Returns a tensor representing a decaying learning rate

    :param str decay:
        type of decay, one of ['constant', 'linear']
    :param float init_lr:
        inital learning rate
    :param tf.Tensor global_step:
        tensor representing a global step, if None will use tensorflow's default
    :param float decay_steps:
        steps over which the learning rate should decay
    :param float end_lr:
        final value of the learning rate

    :rtype: tf.Tensor
    :return:
        tensor representing the decaying learning rate
    """
    global_step = global_step or tf.train.get_global_step() or tf.train.create_global_step()

    try:
        decay_fn = _decay_types[decay]
    except KeyError:
        raise ValueError('Unknown decay type "%s", use one of: %s' % (decay, types()))

    return decay_fn(init_lr, global_step=global_step, decay_steps=decay_steps, end_learning_rate=end_lr)


def types():
    """
    Returns a list of available decay types

    :return:
        list of available decay types
    """
    return list(_decay_types.keys())


_decay_types = {
    'constant': lambda init_lr, *args, **kwargs: init_lr,
    'linear': tf.train.polynomial_decay,
}
