import cv2
import face_alignment
import numpy as np
import skimage.io as sio
from skimage.transform import resize
from torch._C import device


class FaceAligner(object):
    """Face alignment class that expects a picure with a signle face

    """

    def __init__(self, resolution: int, 
                 face_detector: str="blazeface",
                 device: str='cpu') -> None:
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, 
            flip_input=False,
            face_detector=face_detector,
            device=device
        )
        self.resolution = resolution

    def __call__(self, img):
        H, W, *_ = img.shape
        point = self.fa.get_landmarks(self._resize128(img))
        
        front = self._get_position(self.resolution)
        if point:
            shape = np.array(point[0])
            shape = shape / 128
            shape[:, 0] = shape[:, 0] * H
            shape[:, 1] = shape[:, 1] * W
            
            shape = shape[17:]
            M = self._transformation_from_points(np.matrix(shape), np.matrix(front))
            out = cv2.warpAffine(img, M[:2], (256, 256))

            return out
        else:
            return img
        
    def _resize128(self, img):
        return (resize(img, (128, 128)) * 255).astype('uint8')

    def _get_position(self, size, padding=0.25):
        x = [
            0.000213256,
            0.0752622,
            0.18113,
            0.29077,
            0.393397,
            0.586856,
            0.689483,
            0.799124,
            0.904991,
            0.98004,
            0.490127,
            0.490127,
            0.490127,
            0.490127,
            0.36688,
            0.426036,
            0.490127,
            0.554217,
            0.613373,
            0.121737,
            0.187122,
            0.265825,
            0.334606,
            0.260918,
            0.182743,
            0.645647,
            0.714428,
            0.793132,
            0.858516,
            0.79751,
            0.719335,
            0.254149,
            0.340985,
            0.428858,
            0.490127,
            0.551395,
            0.639268,
            0.726104,
            0.642159,
            0.556721,
            0.490127,
            0.423532,
            0.338094,
            0.290379,
            0.428096,
            0.490127,
            0.552157,
            0.689874,
            0.553364,
            0.490127,
            0.42689,
        ]

        y = [
            0.106454,
            0.038915,
            0.0187482,
            0.0344891,
            0.0773906,
            0.0773906,
            0.0344891,
            0.0187482,
            0.038915,
            0.106454,
            0.203352,
            0.307009,
            0.409805,
            0.515625,
            0.587326,
            0.609345,
            0.628106,
            0.609345,
            0.587326,
            0.216423,
            0.178758,
            0.179852,
            0.231733,
            0.245099,
            0.244077,
            0.231733,
            0.179852,
            0.178758,
            0.216423,
            0.244077,
            0.245099,
            0.780233,
            0.745405,
            0.727388,
            0.742578,
            0.727388,
            0.745405,
            0.780233,
            0.864805,
            0.902192,
            0.909281,
            0.902192,
            0.864805,
            0.784792,
            0.778746,
            0.785343,
            0.778746,
            0.784792,
            0.824182,
            0.831803,
            0.824182,
        ]

        x, y = np.array(x), np.array(y)

        x = (x + padding) / (2 * padding + 1)
        y = (y + padding) / (2 * padding + 1)
        x = x * size
        y = y * size
        return np.array(list(zip(x, y)))

    def _transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack(
            [
                np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                np.matrix([0.0, 0.0, 1.0]),
            ]
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fa = FaceAligner(256)
    plt.figure(figsize=(6, 12))
    for i in range(4):
        im = sio.imread(f"data/training_caip_contest/{np.random.randint(500000)}.jpg")
        img = fa(im)

        plt.subplot(4, 2, i * 2 + 1)
        plt.imshow(im)
        plt.subplot(4, 2, i * 2 + 2)
        plt.imshow(img)

    plt.tight_layout()
