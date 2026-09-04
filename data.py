"""Prepare already selected GRD and GLCM features"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def combine_grd_glcm(grd, glcm):
    grd, glcm = np.asarray(grd), np.asarray(glcm)
    if grd.ndim != 3 or glcm.ndim != 3 or grd.shape[:2] != glcm.shape[:2]:
        raise ValueError('GRD and GLCM must be [samples, time, features] with matching samples/time.')
    if np.iscomplexobj(grd) or np.iscomplexobj(glcm):
        raise ValueError('Provide real GRD and GLCM features.')
    x = np.concatenate((grd, glcm), axis=-1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Input contains NaN or infinity; resolve missing data before training.')
    return x


def make_loader(x, y, *, batch_size=64, shuffle=False, seed=42):
    x, y = np.asarray(x), np.asarray(y)
    if np.iscomplexobj(x) or x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('x must be finite, real [samples, time, features].')
    if y.ndim == 2 and y.shape[1] == 2:
        if not np.all((y == 0) | (y == 1)) or not np.all(y.sum(axis=1) == 1):
            raise ValueError('Expected one-hot labels or integer class indices.')
        y = y.argmax(axis=1)
    if y.ndim != 1 or len(y) != len(x) or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be 0=logged, 1=non-logged, one per sample.')
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def get_ratio(data):
    """Notebook feature order: VH, VV, VH - VV."""
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[-1] < 2 or np.iscomplexobj(data):
        raise ValueError('GRD must be real [samples, time, channels] with VH and VV first.')
    return np.stack((data[:, :, 0], data[:, :, 1],
                     data[:, :, 0] - data[:, :, 1]), axis=-1)


def dataPrepare(sl, nonsl, ps=False, *, rng=None):
    """Notebook shuffling and one-hot labels, with real tensors and aligned indices."""
    if ps:
        raise ValueError('Use ps=False for the GRD+GLCM workflow.')
    rng = np.random if rng is None else rng
    if sl.shape != nonsl.shape or sl.ndim != 3:
        raise ValueError('Balanced SL and non-SL arrays must have identical [N,T,D] shapes.')
    number = len(sl)
    non_order = rng.permutation(number)
    x = np.concatenate((sl, nonsl[non_order]), axis=0)
    y = np.concatenate((np.zeros(number, dtype=int), np.ones(number, dtype=int)))
    order = rng.permutation(number * 2)
    # Map back to the original [SL, non-SL] stack, including its first shuffle.
    index = np.concatenate((np.arange(number), number + non_order))[order]
    return (torch.tensor(x[order], dtype=torch.float32),
            torch.tensor(np.eye(2, dtype=np.float32)[y[order]]), index)


def _polygon_split(ids, rng):
    """Match sklearn train_test_split(test_size=0.1, shuffle=True) ordering."""
    n_test = int(np.ceil(len(ids) * 0.1))
    if n_test < 1 or len(ids) - n_test < 1:
        raise ValueError('Not enough polygons for train/validation/test splitting.')
    order = rng.permutation(len(ids))
    return ([ids[i] for i in order[n_test:]], [ids[i] for i in order[:n_test]])


def load_polygon_data(grd_root, glcm_root, *, seed=42, split_file=None):
    """
    Preserve first-minimum class balancing and shared per-polygon shuffling.
    Validation uses 10% of polygons, then test uses 10% of the remainder.
    Returned positions and source indices follow every sample permutation.
    """
    import json
    from pathlib import Path

    grd_root, glcm_root = Path(grd_root), Path(glcm_root)
    ids = sorted(int(p.name.split('_')[-1]) for p in grd_root.iterdir()
                 if p.is_dir() and p.name.startswith('polygon_'))
    if len(ids) < 3:
        raise ValueError('At least three polygon folders are required.')
    if split_file is not None:
        manifest = json.loads(Path(split_file).read_text())
        seed = int(manifest['seed'])
    else:
        manifest = None
    rng = np.random.RandomState(seed)
    polygons = {}
    shape = None
    glcm_count = None
    for index in ids:
        grd_dir = grd_root / f'polygon_{index}'
        glcm_dir = glcm_root / f'polygon_{index}'
        sl = combine_grd_glcm(get_ratio(np.load(grd_dir / 'sl_grd.npy', allow_pickle=False)),
                              np.load(glcm_dir / 'sl_grd_texture.npy', allow_pickle=False))
        nonsl = combine_grd_glcm(get_ratio(np.load(grd_dir / 'nonsl_grd.npy', allow_pickle=False)),
                                 np.load(glcm_dir / 'nonsl_grd_texture.npy', allow_pickle=False))
        sl_pos = np.load(grd_dir / 'sl_s2_pos.npy', allow_pickle=False)
        nonsl_pos = np.load(grd_dir / 'nonsl_s2_pos.npy', allow_pickle=False)
        if len(sl_pos) != len(sl) or len(nonsl_pos) != len(nonsl):
            raise ValueError(f'Position/sample count mismatch in polygon_{index}.')
        if sl.shape[1:] != nonsl.shape[1:] or (shape is not None and sl.shape[1:] != shape):
            raise ValueError(f'Inconsistent time/feature dimensions in polygon_{index}.')
        shape = sl.shape[1:]
        glcm_count = shape[-1] - 3
        number = min(len(sl), len(nonsl))
        if not number:
            raise ValueError(f'Empty class in polygon_{index}.')

        order = rng.permutation(number)
        polygons[index] = dict(sl=sl[order], nonsl=nonsl[order],
                               sl_pos=sl_pos[order], nonsl_pos=nonsl_pos[order], rows=order)
    train_ids, val_ids = _polygon_split(ids, rng)
    train_ids, test_ids = _polygon_split(train_ids, rng)
    split_ids = dict(train=train_ids, val=val_ids, test=test_ids)
    if manifest is not None:
        split_ids = {key: [int(i) for i in manifest['polygon_ids'][key]]
                     for key in ('train', 'val', 'test')}
        flat = [i for values in split_ids.values() for i in values]
        if any(not values for values in split_ids.values()) or sorted(flat) != ids or len(set(flat)) != len(flat):
            raise ValueError('Split file must assign each current polygon exactly once.')
        expected_counts = {str(i): len(polygons[i]['sl']) for i in ids}
        if manifest.get('balanced_samples_per_class') != expected_counts:
            raise ValueError('Polygon sample counts differ from the saved split.')
        if manifest.get('input_shape') != list(shape):
            raise ValueError('Input shape differs from the saved split.')
    result = {}
    for split in ('val', 'test', 'train'):
        selected = split_ids[split]
        sl = np.concatenate([polygons[i]['sl'] for i in selected])
        nonsl = np.concatenate([polygons[i]['nonsl'] for i in selected])
        positions = np.concatenate([
            np.concatenate([polygons[i][key] for i in selected])
            for key in ('sl_pos', 'nonsl_pos')])
        polygon_ids = np.concatenate([np.full(len(polygons[i]['sl']), i) for i in selected])
        rows = np.concatenate([polygons[i]['rows'] for i in selected])
        x, y, order = dataPrepare(sl, nonsl, ps=False, rng=rng)
        result[split] = dict(x=x, y=y, positions=positions[order],
                             polygon_ids=np.tile(polygon_ids, 2)[order],
                             source_rows=np.tile(rows, 2)[order], indices=order)
    result['manifest'] = dict(seed=seed, polygon_ids=split_ids,
                              balanced_samples_per_class={str(i): len(polygons[i]['sl']) for i in ids},
                              input_shape=list(shape),
                              feature_order=['VH', 'VV', 'VH_minus_VV'] +
                                            [f'GLCM_{i}' for i in range(glcm_count)],
                              grd_root=str(grd_root.resolve()), glcm_root=str(glcm_root.resolve()))
    return result
