import numpy as np
import skimage as sk
import skimage.transform as transform
import random

runi = random.uniform


def homogeneous_coords(coords):
    # coords = [x, y] -> Nx2
    N, _ = coords.shape
    homogen_coords = np.concatenate(
        [coords.T, np.ones((1, N))], axis=0)  # x, y, 1 vertical
    return homogen_coords


def nonhomogeneous_coords(coords):
    # coords = 3 x N
    return (coords.T)[:, :2]


def get_valid_transformations(anchor_coords, output_shape, count_tf=200):
    # anchor_coords = anchor homogeneous coordinates to be valid in the resulting transformation, [x, y, 1] vertical
    # output_shape = (h, w)
    # count_tf = number of valid transformations to collect
    print("searching for valid transformations")
    h, w = output_shape
    shift_y, shift_x = np.array([h, w]) / 2.
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.SimilarityTransform(
        translation=[shift_x, shift_y])
    valid_tforms = []

    while len(valid_tforms) < count_tf:

        scale_x, scale_y = (runi(0.25, 0.7), runi(0.25, 0.7))
        translate = None  # (100 - 100 * scale_x, 100 - 100 * scale_x)
        tform = transform.AffineTransform(scale=(scale_x, scale_y), translation=translate,
                                          rotation=runi(0., 3.14), shear=runi(0.2, 0.5))

        final_tform = tf_shift + tform + tf_shift_inv
        warped_coords = np.matmul(final_tform.params, anchor_coords)
        if (np.all(warped_coords[0] >= 0) and np.all(warped_coords[0] < w) and
                np.all(warped_coords[1] >= 0) and np.all(warped_coords[1] < h)):
            valid_tforms.append({'params': final_tform.params, 'coords': warped_coords})
            #valid_tforms.append({'params': np.linalg.inv(final_tform.params), 'coords': warped_coords})                

    return valid_tforms
