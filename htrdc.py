import numpy as np
from skimage.transform import hough_line


def _compute_rd(ru, k):
    """
    Compute rd following formula 6 of the paper.
    :param ru:
    :param k:
    :return:
    """
    x = ru / (2*k)
    y = np.sqrt(np.power((1 / (3*k)), 3) +  np.square(x))
    first = np.cbrt(x + y)
    second = np.cbrt(x - y)
    return first + second


def _compute_undistorted(img, cy, cx, rs, points):
    """
    Compute the 'undistorted' version of the image following formula 7
    :param img:
    :param cy:
    :param cx:
    :param rs:
    :param points:
    :return:
    """
    out = np.zeros_like(img)
    rs = np.vstack((rs, rs)).T
    distorted = np.multiply(points, rs)
    distorted = np.round(distorted).astype(np.int)
    out[points[:, 0] + cy, points[:, 1] + cx] = img[distorted[:, 0] + cy, distorted[:, 1] + cx]
    out[cy, cx] = img[cy, cx]
    return out


def _get_center_and_points(h, w):
    """
    Compute the center of the image and the couple of coordinates of the image.
    The coordinates are inserted in an ndarray with size (h*w, 2) and each row contains the coordinates in
    (row_c, col_c) order.
    :param h:
    :param w:
    :return:
    """
    cx = w // 2
    cy = h // 2
    y_range = np.arange(start=-cy, stop=h - cy, step=1)
    x_range = np.arange(start=-cx, stop=w - cx, step=1)

    points = np.vstack((np.repeat(y_range, w), np.tile(x_range, h))).T
    # Delete the point (0, 0).
    points = np.delete(points, cy*w+cx, axis=0)
    return cy, cx, points


def HTRDC(edges, range_k, n, epsilon):
    """

    :param edges: image after canny edge detector has been applied
    :param range_k: tuple containing initial values for k_min and k_max
    :param n: number of iterations for each distortion iteration
    :param epsilon: acceptable error
    :return:
    """
    h, w = edges.shape
    thetas = np.deg2rad(np.arange(start=0, stop=181, step=1))

    k_min, k_max = range_k

    center_y, center_x, points = _get_center_and_points(h, w)
    ru = np.sqrt(np.sum(np.square(points), axis=1))

    max_k = None

    while (k_max - k_min) > epsilon:
        step = (k_max - k_min) / n
        ks = np.arange(start=k_min + step, stop=k_max, step=step)
        if ks.size < n:
            ks = np.hstack((ks, np.array([k_max], dtype=np.float)))
        rd = []
        for k in ks:
            res = _compute_rd(ru, k)
            rd.append(res)
        rd = np.array(rd, dtype=np.float)
        ru_copy = np.tile(ru, n).reshape(rd.shape)
        rs = rd / ru_copy
        HT_maxs = np.zeros(rs.shape[0], dtype=np.float)
        for i in np.arange(rs.shape[0]):
            und = _compute_undistorted(edges, center_y, center_x, rs[i], points)
            acc, _, _ = hough_line(und, thetas)
            current_max = np.max(acc)
            HT_maxs[i] = current_max
        argmax = np.argmax(HT_maxs)
        max_k = ks[argmax]
        k_min = np.max([0., max_k - step])
        k_max = np.min([max_k + step, 1.])
        n = np.int(np.round(n * 1.10))

    return max_k


def undistort(img, k):
    h, w = img.shape[0], img.shape[1]
    cy, cx, points = _get_center_and_points(h, w)
    ru = np.sqrt(np.sum(np.square(points), axis=1))
    rd = _compute_rd(ru, k)
    r = rd / ru
    return _compute_undistorted(img, cy, cx, r, points)
