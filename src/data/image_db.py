import os
import sys
import io
from pathlib import Path
import time

import lmdb
from PIL import Image
from nuscenes.nuscenes import NuScenes


def image_to_bytes(im):
    byteImgIO = io.BytesIO()
    im.save(byteImgIO, "PNG")
    return byteImgIO.getvalue()

def image_from_bytes(img_bytes):
    return Image.open(io.BytesIO(img_bytes)).convert(mode='RGB')

def get_cam_front_path(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    cam_front_token = sample['data']['CAM_FRONT']
    cam_front_data = nusc.get('sample_data', cam_front_token)
    cam_front_path = Path(nusc.dataroot) / Path(cam_front_data['filename'])
    return cam_front_path


def get_key_value_bytes(nusc, sample_token):
    cam_front_path = get_cam_front_path(nusc, sample_token)

    im = Image.open(str(cam_front_path))
    im_bytes = image_to_bytes(im)

    key_bytes = bytes(cam_front_path.stem, 'utf-8')

    return key_bytes, im_bytes

def write_images_to_lmdb(tokens, nusc, path, map_size):
    start = time.perf_counter()
    print(path)
    total_bytes = 0
    with lmdb.open(path=path, map_size=map_size) as images_db:
        for i, sample_token in enumerate(tokens):
            with images_db.begin(write=True) as txn:
                key, value = get_key_value_bytes(nusc, sample_token)
                total_bytes += len(key) + len(value)
                txn.put(key, value)
            if (i + 1) % 10 == 0:
                print(f'{i + 1} images put into lmdb. {total_bytes / (1024 * 1024 * 1024):.2f} GB written.', int(time.perf_counter()-start), 'sec')

def main_trainval(i):
    start = time.perf_counter()

    data_root = Path.resolve(Path('/N/slate/deduggi/nuScenes-trainval'))
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)
    print('v1.0-trainval loaded', int(time.perf_counter()-start), 'sec')
    
    start = time.perf_counter()
    image_db_path = data_root / Path(f'lmdb/samples/CAM_FRONT_{i+1}')
    image_db_path.mkdir(parents=True, exist_ok=True)
    print('writing to', image_db_path)

    image_db_map_size = int(15 * 1024 * 1024 * 1024)
    sample_tokens = [sample['token'] for sample in nusc.sample][i*7000:(i+1)*7000]
    print('number of samples', len(sample_tokens), int(time.perf_counter()-start), 'sec')
    
    write_images_to_lmdb(sample_tokens, nusc, str(image_db_path), image_db_map_size)

# def main_test():
#     data_root = Path.resolve(Path('/N/slate/deduggi/nuScenes-test'))
#     nusc = NuScenes(version='v1.0-test', dataroot=data_root, verbose=False)
#     image_db_path = data_root / Path('lmdb/samples/CAM_FRONT')
#     image_db_path.mkdir(parents=True, exist_ok=True)
#     image_db_map_size = int(100 * 1024 * 1024 * 1024)
#     sample_tokens = [sample['token'] for sample in nusc.sample]
#     write_images_to_lmdb(sample_tokens, nusc, str(image_db_path), image_db_map_size)

def main_mini():
    data_root = Path.resolve(Path('/N/u/deduggi/Carbonate/autonomous-robotics/translating-images-into-maps/nuscenes_data'))
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)
    image_db_path = data_root / Path('lmdb/samples/CAM_FRONT')
    image_db_path.mkdir(parents=True, exist_ok=True)
    image_db_map_size = int(1 * 1024 * 1024 * 1024)
    sample_tokens = [sample['token'] for sample in nusc.sample]
    write_images_to_lmdb(sample_tokens, nusc, str(image_db_path), image_db_map_size)

if __name__=='__main__':
    # main_mini()
    i = int(sys.argv[1])
    main_trainval(i)