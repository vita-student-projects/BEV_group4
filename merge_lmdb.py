from pathlib import Path

import time
import lmdb
from itertools import islice
def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

data_root = Path.resolve(Path('/N/slate/deduggi/nuScenes-trainval'))
main_db_path = data_root / Path('lmdb/samples/CAM_FRONT')
main_db = lmdb.open(path=str(main_db_path), map_size=int(50 * 1024 * 1024 * 1024))
    
for i in range(4,5):
    image_db_path = data_root / Path(f'lmdb/samples/CAM_FRONT_{i+1}')
    print(image_db_path)
    start = time.perf_counter()
    with lmdb.open(path=str(image_db_path), readonly=True, readahead=False, max_spare_txns=128,) as image_db:
        with image_db.begin() as read_txn:
            cursor = read_txn.cursor()
            for batch in batched(cursor, 10):
                with main_db.begin(write=True) as write_txn:
                    for (key, value) in batch:
                        write_txn.put(key, value)
    print('done', int(time.perf_counter()-start), 'sec')

