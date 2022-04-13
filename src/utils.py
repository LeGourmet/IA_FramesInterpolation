import os
import numpy as np
import tensorflow as tf


def setup_cuda_device(gpus: str = "-1"):
    """ force enable or disable cuda devices"""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Num GPUs Available: ", len(physical_devices))


def get_stitched_image(nb_images, image_array, target_dim_x, target_dim_y):
    """returns a stitched image matching the given dimensions from an array
    Image is scaled up if needed,
    Image cannot be scaled down"""
    array_dim_x, array_dim_y = (image_array[0].shape[0]), (image_array[0].shape[1])
    scale_factor_x = max(target_dim_x // array_dim_x, 1)
    scale_factor_y = max(target_dim_y // array_dim_y, 1)
    stitched = np.reshape(image_array.swapaxes(0, 1), (array_dim_x, array_dim_y * nb_images))
    stitched = np.repeat(stitched, scale_factor_y, axis=0)
    stitched = np.repeat(stitched, scale_factor_x, axis=1)
    return stitched


def grid_images(image_array, nb_row):
    """returns a 2D grid of images of size (nb_row, nb_row)"""
    dim_x, _ = get_image_dimensions(image_array)
    slice_dim = dim_x * nb_row

    grid = np.array(image_array[0:nb_row, 0:dim_x, 0:dim_x])
    grid = grid.reshape(slice_dim, dim_x, 1)

    for i in range(1, nb_row):
        slice = image_array[i * nb_row:(i + 1) * nb_row, 0:dim_x, 0:dim_x]
        slice = slice.reshape(slice_dim, dim_x, 1)
        grid = np.hstack((grid, slice))

    grid = np.reshape(grid, (slice_dim, slice_dim, 1))
    return grid


def get_image_dimensions(Y):
    """returns a tuple containing image dimension from a np image array"""
    dim_x, dim_y = Y[0].shape[0], Y[0].shape[1]
    return dim_x, dim_y


def shuffle_arrays(X, Y):
    """shuffle all X and Y in the dataset.
    Each X always correspond to the same Y after operation"""
    shuffle = np.random.permutation(len(X))
    X = X[shuffle]
    Y = Y[shuffle]
    return X, Y


def get_Z_uniform(size, dimension=1, start=-1, end=1):
    """Generate random latent vectors of size (size, dimension)
    The latent vectors are uniformly distributed in the range [start, end]"""
    return np.random.uniform(start, end, (size, dimension))


def get_Z_normal(size, dimension=1):
    """Generate random latent vectors of size (size, dimension)
    The latent vectors are uniformly distributed in the range [start, end]"""
    return np.random.uniform(-1, 1, (size, dimension))


def get_fake_data(model, size, z=None):
    """Generate fake data of size (size, 1)
    z is generated with normal distribution if not specified"""
    if z is None:
        z = get_Z_uniform(size, model.latent_dim)
    return model.generator(z)


def subImage(batch, size, x, y):
    sX = x-(size//2)
    maxX = len(batch[0])
    sY = y-(size//2)
    maxY = len(batch[0][0])
    res = []
    for i in range(batch.shape[0]):
        res.append([])
        for j in range(sX,sX+size):
            res[-1].append([])
            for k in range(sY,sY+size):
                res[-1][-1].append([0,0,0] if (j<0) | (j>=maxX) | (k<0) | (k>=maxY) else batch[i][j][k])
    return np.array(res)