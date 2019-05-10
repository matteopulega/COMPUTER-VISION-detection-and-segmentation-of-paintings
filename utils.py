from parameters import *

import numpy as np
from scipy.spatial import distance as dist
import cv2


RECTANGLE_SECTOR_TL = 0
RECTANGLE_SECTOR_TR = 1
RECTANGLE_SECTOR_BR = 2
RECTANGLE_SECTOR_BL = 3


def linear_stretch(img, a, b):
    """
    Stretch the image
    :param img: 2D ndarray representing an image plane
    :param a: scale factor
    :param b: sum factor
    :return: The stretched image
    """
    img = img.astype(np.float)
    img = np.round(img * a + b)
    img[img < 0.] = 0.
    img[img > 255.] = 255.
    return img.astype(np.uint8)


def labels_map_to_bgr(labels_map):
    """
    Maps a label map to a bgr image.
    :param labels_map: labels map
    :return: bgr image
    """
    labels_hue = np.uint8(179 * labels_map / labels_map.max())
    blank_ch = 255 * np.ones_like(labels_hue)
    labeled_img = cv2.merge((labels_hue, blank_ch, blank_ch))
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[labels_hue == 0] = np.zeros(3, dtype=np.uint8)
    return labeled_img


def show(img, winname = 'out'):
    """
    Show an image
    :param img: image to show
    :param winname: windows name
    :return:
    """
    cv2.imshow(winname, img)
    cv2.waitKey()


def resize_to_ratio(img, ratio):
    """
    Resize an image according to the given ration
    :param img: Image to be resized
    :param ratio: ratio used to resize the image
    :return: Image resized
    """
    assert ratio > 0, 'ratio_percent must be > 0'
    w = int(img.shape[1] * ratio)
    h = int(img.shape[0] * ratio)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def compute_histogram(img):
    """
    Compute the histogram of the image
    :param img: 2D or 3D array
    :return: normalized histogram
    """
    assert len(img.shape) >= 2, 'img.shape not valid'
    if len(img.shape) == 3:
        h, w, d = img.shape
        h_w = h * w
        if d == 3:
            p1 = img[:, :, 0]
            p2 = img[:, :, 1]
            p3 = img[:, :, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]

    if len(img.shape) == 2:
        h_w, d = img.shape
        if d == 3:
            p1 = img[:, 0]
            p2 = img[:, 1]
            p3 = img[:, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]

    histogram = np.zeros(256*d) # e' corretto 256, non h_w
    for i in np.arange(len(planes)):
        p = planes[i]
        for val in np.unique(p):
            count = np.sum(p == val)
            histogram[val + i*256] = count
    histogram = histogram / img.size
    return histogram


def convert_to(img, flag):
    """
    Convert the image into the given format
    :param img: Image to be converted
    :param flag: Flag pointing to the conversion type
    :return: Converted Image
    """
    return cv2.cvtColor(img, flag)


def entropy(histogram):
    """
    Compute Shannon's Entropy
    :param histogram: Histogram approximation of the real distribution
    :return: Shannon's Entropy
    """
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))


def remove_small_connectedComponents(labeled_img, threshold):
    """
    Function used to remove the small components
    :param labeled_img: Input image with components
    :param threshold: Area threshold
    :return: Image without the small components
    """
    img = np.array(labeled_img)
    labels, counts = np.unique(img, return_counts=True)
    for label, count in zip(labels, counts):
        if count < threshold:
            img[img == label] = 0
    return img


def sort_corners(corners):
    """
    Sort corners starting from top left one and going clockwise
    :param corners:
    :return: sorted corners
    """
    # find the "center"
    c = find_intersection(corners[0], corners[2], corners[1], corners[3])
    if not (min(corners[0][0], corners[2][0]) <= c[0] <= max(corners[0][0], corners[2][0])
        and min(corners[0][1], corners[2][1]) <= c[1] <= max(corners[0][1], corners[2][1])):
        c = find_intersection(corners[0], corners[1], corners[3], corners[2])
        if not (min(corners[0][0], corners[1][0]) <= c[0] <= max(corners[0][0], corners[1][0])
            and min(corners[0][1], corners[1][1]) <= c[1] <= max(corners[0][1], corners[1][1])):
            c = find_intersection(corners[0], corners[3], corners[1], corners[2])
    # get angle with each point
    a0 = find_angle_with_horizontal(c, corners[0])
    a1 = find_angle_with_horizontal(c, corners[1])
    a2 = find_angle_with_horizontal(c, corners[2])
    a3 = find_angle_with_horizontal(c, corners[3])
    idxes = np.argsort([a0, a1, a2, a3])
    return np.array([corners[idxes[2], :], corners[idxes[3], :], corners[idxes[0], :], corners[idxes[1], :]],
                    dtype=np.int)


def find_distance_squared(p1, p2):
    """
    :param p1: [p1x, p1y]
    :param p2: [p2x, p2y]
    :return: Scalar. squared distance between p1 and p2
    """
    assert len(p1) == 2, 'len(p1) must be equal to 2'
    assert len(p2) == 2, 'len(p2) must be equal to 2'
    return ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)


def find_distance(p1, p2):
    """
    :param p1: [p1x, p1y]
    :param p2: [p2x, p2y]
    :return: Scalar. distance between p1 and p2
    """
    return np.sqrt(find_distance_squared(p1, p2))


def find_angle_with_horizontal(a, b, length=None, use_negative=False):
    """
    :param a: [Ax, Ay]
    :param b: [Bx, By]
    :param length: Scalar. if not provided, will be calculated
    :param use_negative: True/False. When set to True, instead of angles greater than PI, negative values will be returned
    :return: Scalar. angle in radiant
    """
    if length is None:
        length = find_distance(a, b)
    assert length > 0, 'distance between points should be greater than 0'
    angle = np.arccos((b[0] - a[0]) / length)
    if a[1] > b[1]:
        if use_negative is True:
            angle = -angle
        else:
            angle = 2*np.pi - angle
    return angle


def radiant_to_degree(angle):
    return angle * 180 / np.pi


def check_if_picture(colored_img, greyscale_img, mask):
    picture = colored_img[mask == 255]
    hist = compute_histogram(picture)
    
    # -- CHECK IF WHITE STATUE
    n = (hist[:256] + hist[256:512] + hist[512:768])
    if np.sum(n[STATUE_GRAY_HIST_THRESH:]) > np.sum(n[:STATUE_GRAY_HIST_THRESH]):
        if DEBUG:
            print("Not picture -> STATUE")
        return False
    
    ent = entropy(hist)
    if ent <= ENTROPY_SURE_NOT_PICTURE_THRESH:
        if DEBUG:
            print('Component is not a picture. Cause: ENTROPY')
        return False
    elif ent >= ENTROPY_SURE_PICTURE_THRESH:
        return True
    else:
        gray_picture = greyscale_img[mask == 255]
        mean = gray_picture.mean()
        if mean < PICTURE_GRAY_THRESH:
            return True
        else:
            if DEBUG:
                print('Component is not a picture. Cause: GRAY')
            return False


def create_non_repeated_couples_of_indexes(n_indexes):
    assert n_indexes > 0, 'n_indexes must be > 0'
    idxs = np.arange(n_indexes)
    idxs = np.vstack((np.repeat(idxs, n_indexes), np.tile(idxs, n_indexes))).T
    idxs = idxs[idxs[:, 0] != idxs[:, 1]]
    idxs = np.sort(idxs, axis=1)
    idxs = np.unique(idxs, axis=0)
    return idxs


def find_intersection(l1_start, l1_end, l2_start, l2_end):
    assert len(l1_start) == 2, 'len(l1_start) must be equal to 2'
    assert len(l1_end) == 2, 'len(l1_end) must be equal to 2'
    assert len(l2_start) == 2, 'len(l2_start) must be equal to 2'
    assert len(l2_end) == 2, 'len(l2_end) must be equal to 2'
    x1, y1 = l1_start
    x2, y2 = l2_start
    x3, y3 = l1_end
    x4, y4 = l2_end

    if x3 == x1:
        x3 = x3 + 1
    if x4 == x2:
        x4 = x4 + 1
    a = (y3 - y1) / (x3 - x1)
    b = (y4 - y2) / (x4 - x2)
    if a == b:
        a = (1 + y3 - y1) / (x3 - x1)
    x = (x1 * a - x2 * b - y1 + y2) / (a - b)
    y = (x - x1) * a + y1
    return np.round([x, y]).astype(np.int)


def find_midpoint(p1, p2):
    assert len(p1) == 2, 'len(p1) must be equal to 2'
    assert len(p2) == 2, 'len(p2) must be equal to 2'
    return np.round([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]).astype(np.int)


def find_biggest_halfrectangle(p1, p2, p3, p4):
    assert len(p1) == 2, 'len(p1) must be equal to 2'
    assert len(p2) == 2, 'len(p2) must be equal to 2'
    assert len(p3) == 2, 'len(p3) must be equal to 2'
    assert len(p4) == 2, 'len(p4) must be equal to 2'
    c = find_intersection(p1, p3, p2, p4)
    v_inf = find_intersection(p1, p4, p2, p3)
    h_inf = find_intersection(p1, p2, p3, p4)
    m12 = find_intersection(p1, p2, c, v_inf)
    m23 = find_intersection(p2, p3, c, h_inf)
    m34 = find_intersection(p3, p4, c, v_inf)
    m41 = find_intersection(p4, p1, c, h_inf)
    bottom_side = False
    left_side = False
    if find_distance_squared(c, m12) < find_distance_squared(c, m34):
        bottom_side = True
    if find_distance_squared(c, m23) < find_distance_squared(c, m41):
        left_side = True
    if bottom_side and left_side:
        return np.array([m41, c, m34, p4], dtype=np.int), RECTANGLE_SECTOR_BL
    elif bottom_side and not left_side:
        return np.array([c, m23, p3, m34], dtype=np.int), RECTANGLE_SECTOR_BR
    elif not bottom_side and left_side:
        return np.array([p1, m12, c, m41], dtype=np.int), RECTANGLE_SECTOR_TL
    else: # not bottom_side and not left_side
        return np.array([m12, p2, m23, c], dtype=np.int), RECTANGLE_SECTOR_TR


def rectify_image(img, corners):
    assert corners.shape == (4, 2), 'corners.shape must be (4, 2)'

    corners = corners.astype(np.float32)
    if DEBUG:
        print("rectify_img -- corners:\n", corners)
    new_vertices = corners.copy()
    for i in range(RECTIFY_IMAGE_ITER):
        new_vertices, sector = find_biggest_halfrectangle(new_vertices[0], new_vertices[1],
                                                          new_vertices[2], new_vertices[3])
    w = int((2 ** RECTIFY_IMAGE_ITER) * max(find_distance(new_vertices[0], new_vertices[1]),
                                            find_distance(new_vertices[2], new_vertices[3])))
    h = int((2 ** RECTIFY_IMAGE_ITER) * max(find_distance(new_vertices[1], new_vertices[2]),
                                            find_distance(new_vertices[3], new_vertices[0])))
    new_vertices = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    if DEBUG:
        print("rectify_img -- new_vertices:\n", new_vertices)
    mat = cv2.getPerspectiveTransform(corners, new_vertices)
    return cv2.warpPerspective(img, mat, (w, h))


def get_opencv_major_version(version):
    major, _, _ = version.split('.')
    return major
