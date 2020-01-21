import glob
from vampire.common.util import save_sparse, load_sparse
from tqdm import trange, tqdm
import os
shard = 10
for jx, file in tqdm(enumerate(glob.glob("1b/preprocessed_shards/*"))):
    z = load_sparse(file)
    if z.shape[0] > 100:
        batch_size = z.shape[0] // shard
        for ix in trange(0, z.shape[0], batch_size):
            if ix + batch_size > z.shape[0]:
                mat = z[ix:,:]
            else:
                mat = z[ix:ix+batch_size,:]
            save_sparse(mat, os.path.join("1b", "preprocessed_shards_1", f"train.{jx}.{ix}.npz"))
    else:
            save_sparse(mat, os.path.join("1b", "preprocessed_shards_1", f"train.{jx}.npz"))
