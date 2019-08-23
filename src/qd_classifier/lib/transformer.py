import json
import numpy as np
from PIL import Image

from qd.tsv_io import TSVDataset
from qd.qd_common import img_from_base64

class ResizeImageWithPadding():
    def __init__(self, size, padding_value=0):
        self.size = size
        self.padding_value = padding_value

    def __call__(self, src_image):
        target_width = self.size
        target_height = self.size
        if src_image.width > src_image.height:
            target_height = int(float(target_width) / src_image.width *
                    src_image.height)
        else:
            target_width = int(float(target_height) / src_image.height *
                    src_image.width)

        resized_image = np.array(src_image.resize((target_width, target_height),
                Image.ANTIALIAS))
        padded_image = np.zeros([self.size, self.size, 3], np.uint8)
        padded_image.fill(self.padding_value)

        xs = 0
        xe = resized_image.shape[1]
        ys = 0
        ye = resized_image.shape[0]

        x_offset = int((self.size - xe) / 2)
        y_offset = int((self.size - ye) / 2)
        xs += x_offset
        xe += x_offset
        ys += y_offset
        ye += y_offset

        padded_image[ys:ye, xs:xe, :] = np.asarray(resized_image,
                dtype=np.uint8)[:, :, 0:3]
        padded_image = Image.fromarray(padded_image)

        return padded_image

class ConvertBGR2RGB(object):
    def __call__(self, src_bgr_image):
        src_B, src_G, src_R = src_bgr_image.getchannel('R'), src_bgr_image.getchannel('G'), src_bgr_image.getchannel('B')
        rgb_image = Image.merge('RGB', [src_R, src_G, src_B])

        return rgb_image

class AddBackground():
    def __init__(self, bg_dataset=None, bg_split='train', bg_version=0,
            use_bg_rect=True, bg_size_ratio=2.0):
        self.bg_dataset = TSVDataset(bg_dataset) if bg_dataset else None
        self.bg_split = bg_split
        self.bg_version = bg_version
        self.use_bg_rect = use_bg_rect
        self.bg_size_ratio = bg_size_ratio
        self.num_images = self.bg_dataset.num_rows(bg_split)

    def __call__(self, src_image):
        if self.bg_dataset is None:
            dst = np.zeros([src_image.height * self.bg_size_ratio,
                    src_image.width * self.bg_size_ratio, 3])
        else:
            dst = None
            while not dst_arr:
                bg_index = np.random.randint(0, self.num_images)
                bg_iter = self.bg_dataset.iter_data(self.bg_split,
                        filter_idx=[bg_index])
                _, _, b64_img = next(bg_iter)
                dst = img_from_base64(b64_img)
                if dst is None:
                    continue

                assert len(dst.shape) == 3
                assert dst.shape[2] == 3
                if self.use_bg_rect:
                    label_iter = self.bg_dataset.iter_data(self.bg_split,
                            t='label', version=self.bg_version,
                            filter_idx=[bg_index])
                    _, str_rects = next(label_iter)
                    rects = json.loads(str_rects)
            # convert BGR to RGB
            dst = dst[:, :, (2,1,0)]

        src = np.array(src_image)
        # randomly put on the background
        dst_start_x = 0
        dst_start_y = 0
        dst_end_x = dst_start_x + src_image.width
        dst_end_y = dst_start_y + src_image.height
        foreground_weight = src[:,:,3] / 255.0
        background_weight = np.ones(src.shape[:2]) - foreground_weight

        for c in range(3):
            dst[dst_start_y:dst_end_y, dst_start_x:dst_end_x, c] = \
                foreground_weight * src[0:src.shape[0], 0:src.shape[1], c] + \
                background_weight * dst[dst_start_y:dst_end_y, \
                dst_start_x:dst_end_x, c]

        patch = Image.fromarray(dst.astype(np.uint8))
        return patch
