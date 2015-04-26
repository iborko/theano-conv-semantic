import logging
import numpy
from sys import maxint, stdout
from datetime import datetime
from util import try_pickle_dump


logger = logging.getLogger(__name__)


def set_layers_training_mode(layers, mode):
    """
    Sets training mode in layers which support traning_mode
    """
    for i, layer in enumerate(layers):
        if 'training_mode' in layer.__dict__:
            # logger.info('Found layer with trainig mode %d, setting to %d',
            #              i, mode)
            layer.training_mode.set_value(mode)


def eval_model(conf, train_fn, test_fn, n_train_batches, n_test_batches,
               layers, pre_fn=None, l_rate_wrapper=None):
    """
    Function for trainig and validating models

    n_epochs: dictionary
        configuration params
    train_fn: theano function
        training function
    test_fn: theano function
        validation function
    n_train_batches: int
        number of batches for training
    n_test_batches: int
        number of batches for validation
    layers: list
        list of layers, used to extract best params
    pre_fn: function
        function to be called before training
    l_rate_wrapper: UpdateParameters object
        learning rate wrapper object

    returns: (best_validation_error, best_iter, best_params)
        the best validation error, iteration and parameters
    """
    assert(type(conf) is dict)
    n_epochs = conf['epochs']
    if n_epochs <= 0:
        n_epochs = maxint

    # how often to lower learning rate if no improvement
    improvement_check_best_loss = numpy.inf
    epochs_check_learn_rate  = None
    if 'learning-rate-decrease-params' in conf:
        lrdp_params = conf['learning-rate-decrease-params']
        epochs_check_learn_rate = lrdp_params['no-improvement-epochs']

    # file for dumping weights
    now = datetime.now()
    weights_filename = "network-%d-%d.bin" % (now.hour, now.minute)

    logger.info('... training')

    #   early-stopping parameters
    # look as this many iterations regardless
    patience = 1200
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.997
    # go through this many minibatche before checking the network
    # on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)
    logger.debug('Validation frequency is %d' % validation_frequency)

    set_layers_training_mode(layers, 1)

    best_validation_loss = numpy.inf
    best_iter = 0
    best_params = []

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        # function to be called before every epoch
        if pre_fn is not None:
            pre_fn()

        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            cost_ij = train_fn(minibatch_index)
            # logger.info('training @ iter = %d, cost %f' % (iter, cost_ij))
            stdout.write('.')
            stdout.flush()

            if (iter + 1) % validation_frequency == 0:
                stdout.write('\n')  # newline after iteration dots

                set_layers_training_mode(layers, 0)

                # compute zero-one loss on validation set
                validation = [test_fn(i) for i in xrange(n_test_batches)]
                validation_losses = [v[0] for v in validation]
                validation_costs = [v[1] for v in validation]
                validation_class_errors = [v[2] for v in validation]

                set_layers_training_mode(layers, 1)

                this_validation_loss = numpy.mean(validation_losses)
                logger.info('epoch %i, minibatch %i/%i, validation error %f %%',
                            epoch, minibatch_index + 1, n_train_batches,
                            this_validation_loss * 100.)
                logger.info('validation cost: %f',
                            numpy.mean(validation_costs))
                logger.info('mean class error: %f %%',
                            numpy.mean(validation_class_errors) * 100.)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # save model parameters
                    best_params = [l.get_weights() for l in layers]
                    try_pickle_dump(best_params, weights_filename)

                    logger.info(('     epoch %i, minibatch %i/%i,'
                                 'test error of best model %f %%') %
                                (epoch, minibatch_index + 1, n_train_batches,
                                 this_validation_loss * 100.))

                # lower learning rate if no improvement
                if epoch % epochs_check_learn_rate:
                    if best_validation_loss >= improvement_check_best_loss:
                        l_rate_wrapper.lower_rate_by_factor(2)
                    improvement_check_best_loss = best_validation_loss

            if patience <= iter:
                logger.info("Run out of patience")
                done_looping = True
                break

    logger.info('Optimization complete.')

    return best_validation_loss, best_iter, best_params
