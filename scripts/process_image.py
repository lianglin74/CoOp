import numpy as np
import cv2
import os
from qd_common import ensure_directory
from qd_common import network_input_to_image
import matplotlib.pyplot as plt
from random import random
import logging

def put_text(im, text, bottomleft=(0,100), 
        color=(255,255,255), font_scale=0.5, 
        font_thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im,text,bottomleft,
            font,font_scale, color,
            thickness=font_thickness)
    return cv2.getTextSize(text, font, font_scale, font_thickness)[0]

def show_net_input(data, label, max_image_to_show=None):
    all_image = network_input_to_image(data, [104, 117, 123])
    num_image = label.shape[0]
    if max_image_to_show:
        num_image = min(max_image_to_show, num_image)
    num_rect = label.shape[1] / 5
    im_height = all_image[0].shape[0]
    im_width = all_image[0].shape[1]
    for i in range(num_image):
        rects = []
        txts = []
        for j in range(num_rect):
            if label[i, j * 5] == 0:
                break
            cx, cy, w, h = label[i, (j * 5 + 0) : (j * 5 + 4)]
            txt = str(label[i, j * 5 + 4])
            cx = cx * (im_width - 1)
            cy = cy * (im_height - 1)
            w = w * (im_width - 1)
            h = h * (im_height - 1)
            rects.append((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 *
                h))
            txts.append(txt)
        draw_bb(all_image[i], rects, txts)
        show_image(all_image[i])

def draw_bb(im, all_rect, all_label, 
        probs=None, 
        color=None,
        #font_scale=0.5, 
        #font_thickness=1,
        #rect_thickness=2,
        draw_label=True):
    '''
    all_rect: x0, y0, x1, y1
    '''
    ref = sum(im.shape[:2]) / 2

    font_scale = ref / 500.
    font_thickness = max(ref / 250, 1)
    rect_thickness = max(ref / 250, 1)

    dist_label = set(all_label)
    if color is None:
        color = {}
        import qd_const
        color = qd_const.label_to_color
        for l in dist_label:
            if l in color:
                continue
            if len(qd_const.gold_colors) > 0:
                color[l] = qd_const.gold_colors.pop()
    for i, l in enumerate(dist_label):
        if l in color:
            continue
        color[l] = (random() * 255., random() * 255, random() * 255)

    if type(all_rect) is list:
        assert len(all_rect) == len(all_label) 
    elif type(all_rect) is np.ndarray:
        assert all_rect.shape[0] == len(all_label)
        assert all_rect.shape[1] == 4
    else:
        assert False
    
    for i in range(len(all_label)):
        rect = all_rect[i]
        label = all_label[i]
        cv2.rectangle(im, (int(rect[0]), int(rect[1])), 
                (int(rect[2]), int(rect[3])), color[label], 
                thickness=rect_thickness)
        if probs is not None:
            if draw_label:
                label_in_image = '{}-{:.2f}'.format(label, probs[i])
            else:
                label_in_image = '{:.2f}'.format(probs[i])
        else:
            if draw_label:
                label_in_image = '{}'.format(label)

        def gen_candidate():
            # above of top left
            yield int(rect[0]) + 2, int(rect[1]) - 4
            # below of bottom left
            yield int(rect[0]) + 2, int(rect[3]) + text_height + 2
        if draw_label or probs is not None:
            (_, text_height), _ = cv2.getTextSize(label_in_image, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_thickness)
            for text_left, text_bottom in gen_candidate():
                if text_left < im.shape[1] - 12 and \
                        text_left >= 0 and \
                        text_bottom > 12 and \
                        text_bottom < im.shape[0]:
                    put_text(im, 
                            label_in_image, 
                            (text_left, text_bottom), 
                            color[label],
                            font_scale,
                            font_thickness)
                    break

def save_image(im, file_name):
    ensure_directory(os.path.dirname(file_name))
    cv2.imwrite(file_name, im)

def load_image(file_name):
    return cv2.imread(file_name)

def show_image(im):
    name = 'image'
    #cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,1)
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range(1, 5):
        cv2.waitKey(1)

def show_images(all_image, num_rows, num_cols):
    _, ax = plt.subplots(num_rows, num_cols)
    ax = ax.reshape((num_rows, num_cols))
    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if k >= len(all_image):
                break
            ax[i, j].imshow(cv2.cvtColor(all_image[k],
                cv2.COLOR_BGR2RGB))
            k = k + 1
    plt.show()
    plt.close()

