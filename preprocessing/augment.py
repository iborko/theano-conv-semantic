import numpy as np
import skimage.transform


default_augmentation_params = {
    'zoom_range': (1.0, 1.1),
    'rotation_range': (0, 10),
    'shear_range': (0, 5),
    'translation_range': (-20, 20),
}


def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0,0), image_shape=(42, 42)):

    tform_augment = skimage.transform.AffineTransform(
        scale=(1/zoom, 1/zoom),
        rotation=np.deg2rad(rotation),
        shear=np.deg2rad(shear),
        translation=translation)

    center_shift = np.array(image_shape)[::-1] / 2. - 0.5
    tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

    # shift to center, augment, shift back (for the rotation/shearing)
    tform = tform_center + tform_augment + tform_uncenter
    return tform


def fast_warp_grayscale(img, tf, output_shape=(42, 42), mode='reflect'):
    """
    This wrapper function is about five times faster than
    skimage.transform.warp, for our use case.
    """
    m = tf.params
    img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')

    img_wf = skimage.transform._warps_cy._warp_fast(
        img, m, output_shape=output_shape, mode=mode)

    return img_wf


def fast_warp_rgb(img, tf, output_shape=(42, 42), mode='reflect'):
    """
    This wrapper function is about five times faster than
    skimage.transform.warp, for our use case.
    """
    m = tf.params
    img_wf = np.empty((img.shape[0], output_shape[0], output_shape[1]),
                      dtype='float32')

    for k in xrange(img.shape[0]):
        img_wf[k, ...] = skimage.transform._warps_cy._warp_fast(
            img[k, ...],
            m,
            output_shape=output_shape,
            mode=mode)

    return img_wf


def random_perturbation_params(
        zoom_range, rotation_range, shear_range, translation_range,
        do_flip=True):

    # random shift
    shift_x, shift_y = 0, 0
    if np.random.randint(2) > 0:
        shift_x = np.random.uniform(*translation_range)
    if np.random.randint(2) > 0:
        shift_y = np.random.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = 0
    if np.random.randint(2) > 0:
        rotation = np.random.uniform(*rotation_range)

    # random shear [0, 5]
    shear = 0
    if np.random.randint(2) > 0:
        shear = np.random.uniform(*shear_range)

    # # flip
    if do_flip and (np.random.randint(2) > 0):  # flip half of the time
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    # random zoom [0.9, 1.1]
    # zoom = np.random.uniform(*zoom_range)
    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range))
    # for a zoom factor this sampling approach makes more sense.
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1]
    # instead of [0.9, 1.1] makes more sense.
    return {'zoom': zoom, 'rotation': rotation, 'shear': shear,
            'translation': translation}


def _perturb_image(img, transformation, output_shape):
    """
    Perturb one image.
    """
    if img.ndim == 2:  # if image is grayscale (1 channel)
        return fast_warp_grayscale(
            img, transformation, output_shape, mode='reflect')
    elif img.ndim == 3:  # if image is color (3 channels)
        return fast_warp_rgb(
            img, transformation, output_shape, mode='reflect')
    else:
        return None


def _div(tup, i):
    ''' Divide tuple with int '''
    return tuple([x / i for x in tup])


def perturb_image(
        img,
        output_shape,
        augmentation_params=default_augmentation_params):

    if len(output_shape) != 2:
        raise RuntimeError("Output shape must have 2 dims.")

    perturb_params = random_perturbation_params(**augmentation_params)
    perturb_params['image_shape'] = output_shape
    # print "\nOriginal params", perturb_params

    if type(img) is tuple or type(img) is list:
        out_list = []
        for i in img:
            #   img size compared to first image
            scale = int(round(1.0 * img[0].shape[-1] / i.shape[-1]))
            img_perturb_params = perturb_params.copy()
            img_perturb_params['translation'] = _div(
                img_perturb_params['translation'], scale)
            img_perturb_params['image_shape'] = _div(output_shape, scale)
            # print "new params", img_perturb_params
            tform_augment = build_augmentation_transform(**img_perturb_params)
            out_shp = img_perturb_params['image_shape']

            out_list.append(_perturb_image(i, tform_augment, out_shp))
        return out_list
    else:
        tform_augment = build_augmentation_transform(**perturb_params)
        return _perturb_image(img, tform_augment, output_shape)
