import multiprocessing as mp
import os

import cv2
import dlib
import numpy as np
import pandas as pd
import tqdm

print("Num CPUs: ", mp.cpu_count())

ALIGNED = False

DLIB_MODEL = 'weights/dlib_face_recognition_resnet_model_v1.dat'
IMG_DIR = 'data/training_caip_contest'
IMG_CSV = 'data/training_caip_contest.csv'
OUT_PATH = 'data/features_dlib.npy'

# Data
df = pd.read_csv(IMG_CSV, header=None)
im_names = df[0].tolist()

N = len(im_names)
acc = np.zeros((N, 128))

# Model
model = dlib.face_recognition_model_v1(DLIB_MODEL)

def get_features(im_name):
    im_path = os.path.join(IMG_DIR, im_name)

    im = cv2.imread(im_path)
    im = cv2.resize(im, (150, 150))

    out = model.compute_face_descriptor(im)
    out = np.array(out)
    return int(im_name.split('.')[0]), out

with mp.Pool(mp.cpu_count()) as pool:
    print('Pool started')
    results = list(tqdm.tqdm(pool.imap(get_features, im_names), total=N))
    
print('Sorting features')
results.sort(key = lambda x: x[0])
results = np.stack([x for _, x in results])

print('Saving')
np.save(OUT_PATH, results)


    
    
