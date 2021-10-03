import os
import time
from multiprocessing import Pool

import pandas as pd
import skimage.io as sio
import tqdm

import config
from align import FaceAligner

skipped = 0

def process_image(impath):
    global skipped
    im = sio.imread(impath)
    im, success = aligner(im)
    if not success:
        skipped += 1
        
        print(f"Skipped: {impath}")
    else:
        op = impath.replace('training_caip_contest', 'training_caip_contest_aligned')
        sio.imsave(op, im)


if __name__ == '__main__':
    df = pd.read_csv('data/training_caip_contest.csv', header=None)
    df.columns = ["image", "label"]
    add_img_dir = lambda x: os.path.join('data/training_caip_contest', x)
    df.image = df.image.apply(add_img_dir)
    
    t1 = time.time()
    aligner = FaceAligner(256)
    # with Pool(20) as p:
    #     p.map(process_image, df.image[:100].tolist())
        
    for impath in tqdm.tqdm(df.image):
        process_image(impath)
        
    print(f"Time: {time.time() - t1}")
    print(f"Skipped files: {skipped}")

    

