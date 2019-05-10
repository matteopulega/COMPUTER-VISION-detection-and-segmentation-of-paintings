from utils import *
from htrdc import HTRDC, undistort
from components import Component
from parameters import *
import os
import sys
import argparse
import glob
import errno
import numpy as np
import cv2


def resize_when_too_big(img, threshold_w_h):
    h = int(img.shape[0])
    w = int(img.shape[1])
    thr_w, thr_h = threshold_w_h
    if h > thr_h or w > thr_h:
        h_ratio = thr_h / h
        w_ratio = thr_w / w
        ratio = min(h_ratio, w_ratio)
        img = resize_to_ratio(img, ratio)
    return img


def read_undistorted_image_color_grayscale(img_file):
    img = cv2.imread(img_file)
    img = resize_when_too_big(img, PICTURE_SIZE_THRESH_W_H)
    gray = convert_to(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLURRING_GAUSSIAN_KERNEL_SIZE, BLURRING_GAUSSIAN_SIGMA)
    edges = cv2.Canny(gray, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    k = HTRDC(edges, (HTRDC_K_START, HTRDC_K_END), HTRDC_N, HTRDC_EPSILON)
    img = undistort(img, k)
    gray = convert_to(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def erode_dilate(img):
    img = cv2.erode(img, np.ones((3, 3), dtype=np.uint8))
    img = cv2.dilate(img, np.ones(DILATE_KERNEL_SIZE, dtype=np.uint8), iterations=DILATE_ITERATIONS)
    img = cv2.erode(img, np.ones(EROSION_KERNEL_SIZE, dtype=np.uint8), iterations=EROSION_ITERATIONS)
    return img


def draw_border_for_picture_parts(drawing):
    flag = False

    sm_column = np.sum(drawing, axis=0)
    if sm_column[0] > 0:
        drawing[:, :5] = 0
        flag = True
    if sm_column[-1] > 0:
        drawing[:, -5:] = 0
        flag = True

    sm_row = np.sum(drawing, axis=1)
    if sm_row[0] > 0:
        drawing[:5, :] = 0
        flag = True
    if sm_row[-1] > 0:
        drawing[-5:, :] = 0
        flag = True

    return drawing, flag


def image_segmentation(gray):
    """
    :param gray: grayscale image
    :return: components, processed grayscale image
    """
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                ADAPTIVE_THRESHOLD_KERNEL_SIZE, ADAPTIVE_THRESHOLD_C)
    gray = cv2.medianBlur(gray, 3)
    gray = erode_dilate(gray)

    _, labeled_img = cv2.connectedComponentsWithAlgorithm(gray, 8, cv2.CV_32S, cv2.CCL_GRANA)
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]
    intermediate_global_mask = np.zeros_like(labeled_img, dtype=np.uint8)
    for label in labels:
        mask = np.zeros_like(labeled_img, dtype=np.uint8)
        mask[labeled_img == label] = 255

        # Compute the convex hull
        if get_opencv_major_version(cv2.__version__) in ['2', '3']:
            mask, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = []
        for cnt in contours:
            hull.append(cv2.convexHull(cnt, False))
        hull_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(len(contours)):
            hull_mask = cv2.drawContours(hull_mask, hull, i, 255, -1, 8)

        intermediate_global_mask = np.clip(intermediate_global_mask + hull_mask, 0, 255)
    return connected_components_segmentation(intermediate_global_mask), gray


def connected_components_segmentation(intermediate_global_mask):
    """
    :param intermediate_global_mask: black and white image
    :return: components
    """
    _, labeled_img = cv2.connectedComponentsWithAlgorithm(intermediate_global_mask, 8, cv2.CV_32S, cv2.CCL_GRANA)
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]

    components = []

    for label in labels:
        mask = np.zeros_like(labeled_img, dtype=np.uint8)
        mask[labeled_img == label] = 255

        # Compute the convex hull
        if get_opencv_major_version(cv2.__version__) in ['2', '3']:
            mask, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = []
        for cnt in contours:
            hull.append(cv2.convexHull(cnt, False))
        hull_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(len(contours)):
            hull_mask = cv2.drawContours(hull_mask, hull, i, 255, -1, 8)

        single_component, flag = draw_border_for_picture_parts(hull_mask)

        _, connected_component, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(single_component, 8,
                                                                                          cv2.CV_32S, cv2.CCL_GRANA)
        valid_labels = np.argwhere(stats[:, cv2.CC_STAT_AREA] >= LABEL_AREA_THRESHOLD)
        if valid_labels[0] == 0:
            valid_labels = valid_labels[1:]
        for valid_label in valid_labels:
            component = Component(valid_label, connected_component, stats[valid_label], flag)
            components.append(component)

    components.sort(key=lambda x: x.area, reverse=True)
    return components


def show_vertices(img, image_vertices, with_order=True):
    print("vertices:\n",image_vertices)
    img_c = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)] #B G Y R
    for i in range(len(image_vertices)):
        vertex = tuple(image_vertices[i])
        img_c = cv2.circle(img_c, vertex, 4, colors[i], thickness=-5-i)
    show(img_c)


def show_rectangle(img, sorted_vertices):
    img_lines = img.copy()
    cv2.line(img_lines , tuple(sorted_vertices[0,:]), tuple(sorted_vertices[1,:]), (0, 255, 0), 3)
    cv2.line(img_lines , tuple(sorted_vertices[1,:]), tuple(sorted_vertices[2,:]), (0, 255, 0), 3)
    cv2.line(img_lines , tuple(sorted_vertices[2,:]), tuple(sorted_vertices[3,:]), (0, 255, 0), 3)
    cv2.line(img_lines , tuple(sorted_vertices[3,:]), tuple(sorted_vertices[0,:]), (0, 255, 0), 3)
    show(img_lines)


def rect(img, mask):
    img_parts = np.copy(img)
    x, y, w, h = cv2.boundingRect(mask)
    cv2.rectangle(img_parts, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show(img_parts, 'Picture part')


def segmentation(img_segm, component):
    if component.picture_part_flag is False:
        img_segm[component.mask == 255] = SEGMENTATION_COLOR_RP
    else:
        img_segm[component.mask == 255] = SEGMENTATION_COLOR_PP
    return img_segm


def extract_picture_parts(img, component):
    x, y, w, h = cv2.boundingRect(component.mask)
    part = img[y:y+h,x:x+w]
    return part


def save_img(out_img, out_file_name):
    print('Saving ', out_file_name)
    return cv2.imwrite(out_file_name, out_img)


def overlap(img, segmentation_mask, component_color, out_color):
    mask = segmentation_mask == component_color
    if np.sum(mask) == 0:
        return img
    mask = np.all(mask, axis=2)
    img2 = np.zeros_like(img, dtype=np.uint8)
    img2[mask] = out_color
    img[mask] = cv2.addWeighted(img[mask], 0.8, img2[mask], 0.2, 1)
    return img


def get_only_file_name(img_file):
    sep = img_file.split(os.path.sep)
    if len(sep) == 1:
        sep = img_file.split('/')
    file = sep[-1]
    file = file.split('.')[0]
    return file


def main(img_file_name, out_dir):
    filename = img_file_name
    img_file_name = get_only_file_name(filename)
    print('Starting processing image ', filename)
    img, gray = read_undistorted_image_color_grayscale(filename)
    if DEBUG is True:
        show(img, img_file_name)
    out_folder = out_dir + '/' + img_file_name
    try:
        os.makedirs(out_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('There were some problem during the creation of folder ', out_folder, '. Skipping image ', img_file_name)
            return
    gray = cv2.GaussianBlur(gray, BLURRING_GAUSSIAN_KERNEL_SIZE, BLURRING_GAUSSIAN_SIGMA)
    components, gray = image_segmentation(gray)
    global_mask = np.zeros_like(gray, dtype=np.uint8)

    img_segm = np.zeros_like(img)
    img_segm[:, :] = SEGMENTATION_COLOR_BG
    i = 0

    for component in components:
        is_contained, global_mask = component.check_if_contained_in_another_component(global_mask)

        if DEBUG is True:
            show(component.mask, 'mask component')

        if is_contained is True:
            continue
        if check_if_picture(img, gray, component.mask) is False:
            continue
        else:
            global_mask[component.mask == 255] = 255

        image_vertices, real_vertices = component.get_vertices(gray)
        if image_vertices is None:
            continue

        if len(image_vertices) == 4:
            if DEBUG is True:
                show_vertices(img, image_vertices, with_order=True)
            sorted_vertices = sort_corners(image_vertices)
            if DEBUG is True:
                show_vertices(img, sorted_vertices, with_order=True)

            if DEBUG is True:
                show_rectangle(img, sorted_vertices)

            img_segm = segmentation(img_segm, component)

            final = rectify_image(img, sorted_vertices)
            if final is not None and component.picture_part_flag is False:
                if DEBUG is True:
                    show(final, 'Regular picture')
                save_img(final, out_folder + '/' + img_file_name + '_painting_' + str(i) + '.jpg')

            if component.picture_part_flag is True:
                if DEBUG is True:
                    rect(img, component.mask)
                p_part = extract_picture_parts(img, component)
                if DEBUG is True:
                    show(p_part,'Picture part')
                save_img(p_part, out_folder + '/' + img_file_name + '_painting_parts_' + str(i) + '.jpg')
            i += 1

    if DEBUG is True:
        show(img_segm, 'Segm')
    segmented_img = img.copy()
    segmented_img = overlap(segmented_img, img_segm, SEGMENTATION_COLOR_RP, SEGMENTATION_COLOR_RP_OUT)
    segmented_img = overlap(segmented_img, img_segm, SEGMENTATION_COLOR_PP, SEGMENTATION_COLOR_PP_OUT)
    save_img(segmented_img, out_folder + '/' + img_file_name + '_segmentation_result.jpg')
    print('End processing image ', filename)



if __name__ == '__main__':
    folder = './test_images'
    images = [img for img in os.listdir(folder)]
    images = sorted(images)
    for name in images:
        print('\n------- START --------')
        #name = 'aaa1.jpg'
        name = '260px-The_Scream.jpg'
        print(name)
        main('{}/{}'.format(folder, name), 'output')
        print('\n------- END --------\n\n')

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input directory containing ONLY image files.', required=False)
    parser.add_argument('--input_img', type=str, help='Input image file.', required=False)
    parser.add_argument('--out_dir', type=str, help='Output directory used to store the results.', required=True)

    args = parser.parse_args()
    if args.input_dir is None and args.input_img is None:
        print('You must specify either an input directory or an input image.')
        sys.exit(-1)
    else:
        if args.input_dir is not None:
            if args.input_dir[-1] == '/':
                args.input_dir = args.input_dir[:-1]
            img_file_list = glob.glob(args.input_dir + '/**')
        else:
            img_file_list = [args.input_img]
    if args.out_dir[-1] == '/':
        args.out_dir = args.out_dir[:-1]
    try:
        os.makedirs(args.out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('There were some problem accessing the output folder.')
            sys.exit(-1)
    for img_file in img_file_list:
        main(img_file, args.out_dir)
'''