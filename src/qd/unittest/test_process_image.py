import base64
import os.path as op
import unittest

class TestProcessImage(unittest.TestCase):
    def test_url_to_3_channel_image(self):
        from qd.qd_common import image_url_to_bytes, encoded_from_img
        from qd.process_image import bytes_to_img_array

        urls = [
                "https://cdn-images-1.medium.com/max/1600/0*akL0KXb54mViVajR.", #gif
                "http://media2.intoday.in/indiatoday/images/stories/seagate-logo_559_063016092548.jpg", #webp
                "https://cdn.pixabay.com/photo/2016/04/20/21/17/png-1342113__340.png", # png
                "https://upload.wikimedia.org/wikipedia/commons/4/41/Sunflower_from_Silesia2.jpg", # jpg
                "http://cdn.shoppers-bay.com/img/a5fc29c9c0f440d84f1ba7845c47e6b8.jpeg", # jpg grayscale
                ]

        for idx, url in enumerate(urls):
            img_bytes = image_url_to_bytes(url)
            imarr = bytes_to_img_array(img_bytes, check_channel=True)
            if imarr is None:
                import ipdb; ipdb.set_trace()
            h, w, c = imarr.shape
            self.assertEqual(c, 3)
            # for debug, visually check loaded image
            # encoded_img = encoded_from_img(imarr)
            # with open(op.join(op.expanduser("~"), "temp/{}.jpg".format(idx)), 'wb') as fp:
            #     fp.write(base64.b64decode(encoded_img))

if __name__ == '__main__':
    unittest.main()