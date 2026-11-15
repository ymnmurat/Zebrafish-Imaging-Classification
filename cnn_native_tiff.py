"""Train and evaluate the native-TIFF zebrafish clot-pattern CNN."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WIDTH, CROP_HEIGHT, FULL_HEIGHT = 240, 96, 180
CROP_UM_PER_PIXEL = 4.0


def decode_12bit_tiff(raw):
    """Read the DP73's native 12-bit samples stored in a 16-bit RGB TIFF."""
    with Image.open(io.BytesIO(raw)) as image:
        tags = image.tag_v2
        if tags[258] != (16, 16, 16) or tags[259] != 1 or tags[284] != 1 or tags[274] != 1:
            raise ValueError('Expected a top-left, uncompressed, interleaved 16-bit RGB TIFF.')
        order = '<' if raw[:2] == b'II' else '>'
        strips = [np.frombuffer(raw, dtype=order+'u2', count=count//2, offset=offset)
                  for offset, count in zip(tags[273], tags[279])]
        size = image.height * image.width * 3
        pixels = np.concatenate(strips)[:size].reshape(image.height, image.width, 3)
    if pixels.max() > 4095:
        raise ValueError('Native TIFF samples exceed the verified DP73 12-bit range.')
    return pixels


def edge_pixels(green):
    """Outer 20 pixels of a 224 × 224 image, with corners counted once."""
    return np.concatenate([green[:20].ravel(), green[-20:].ravel(),
                           green[20:-20,:20].ravel(), green[20:-20,-20:].ravel()])


def background_level(green):
    """Estimate additive green haze from dim pixels around the field of view."""
    return float(np.percentile(edge_pixels(green), 20))


def resize(green, width, height):
    return np.asarray(Image.fromarray(green).resize((width,height),Image.Resampling.BILINEAR))


def tail_axis(green, native_um):
    """Estimate the tail axis from diffuse local texture with capped weights."""
    height,width = green.shape
    factor = 224/max(height,width)
    reduced = resize(green,round(width*factor),round(height*factor))
    pitch = native_um/factor
    fine = ndi.gaussian_filter(reduced,max(.65,4/pitch))
    broad = ndi.gaussian_filter(reduced,max(2.2,28/pitch))
    texture = ndi.gaussian_filter(np.abs(fine-broad),max(1.0,12/pitch))
    low,high = np.percentile(texture,[70,97])
    weight = np.maximum(np.minimum(texture,high)-low,0)
    # Downweight frame boundaries and very intense point-like deposits.
    weight[:3,:] = weight[-3:,:] = weight[:,:3] = weight[:,-3:] = 0
    y,x = np.indices(reduced.shape,dtype=np.float32)
    # Pixel centers converted back to the original raster coordinate system.
    y = (y+.5)*(height/reduced.shape[0])-.5
    x = (x+.5)*(width/reduced.shape[1])-.5
    mass = float(weight.sum())
    if mass <= 1e-8:
        return np.array([(width-1)/2,(height-1)/2]), np.array([1.0,0.0]), 1.0
    center = np.array([(x*weight).sum()/mass,(y*weight).sum()/mass])
    dx,dy = x-center[0],y-center[1]
    covariance = np.array([[(dx*dx*weight).sum(),(dx*dy*weight).sum()],
                           [(dx*dy*weight).sum(),(dy*dy*weight).sum()]])/mass
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:,-1]
    if axis[0]<0:
        axis = -axis
    return center, axis, float(values[-1]/max(values[0],1e-6))


def sample_crop(green, row):
    """Sample a 960 × 384 µm window from the original TIFF plane."""
    center,axis,_ = tail_axis(green,row['um_per_pixel'])
    normal = np.array([-axis[1],axis[0]])
    step = CROP_UM_PER_PIXEL/row['um_per_pixel']
    xx = (np.arange(WIDTH,dtype=np.float32)-(WIDTH-1)/2)*step
    yy = (np.arange(CROP_HEIGHT,dtype=np.float32)-(CROP_HEIGHT-1)/2)*step
    x = center[0]+xx[None,:]*axis[0]+yy[:,None]*normal[0]
    y = center[1]+xx[None,:]*axis[1]+yy[:,None]*normal[1]
    valid = (x>=0)&(x<=green.shape[1]-1)&(y>=0)&(y<=green.shape[0]-1)
    filtered = ndi.gaussian_filter(green,max(.5,.38*step))
    crop = ndi.map_coordinates(filtered,(y,x),order=1,mode='constant',cval=row['green_offset'],prefilter=False)
    return crop, valid


def sample_full(green, level):
    height,width = green.shape
    factor = min(WIDTH/width,FULL_HEIGHT/height)
    h,w = round(height*factor),round(width*factor)
    top,left = (FULL_HEIGHT-h)//2,(WIDTH-w)//2
    full = np.full((FULL_HEIGHT,WIDTH),level,dtype=np.float32)
    valid = np.zeros_like(full,dtype=bool)
    full[top:top+h,left:left+w] = resize(green,w,h)
    valid[top:top+h,left:left+w] = True
    return full,valid


def channels(green, valid, row):
    # One contrast scale per source image, shared by the full and cropped views.
    level = row['green_offset']
    scale = max(8,row['bright_green_p999']-level)
    signal = np.maximum(0,green-level)/scale
    local = signal-ndi.gaussian_filter(signal,2.0)
    return np.stack([np.arcsinh(signal),np.arcsinh(local),valid],axis=0).astype(np.float32)


class Inputs(Dataset):
    def __init__(self, cache, entries):
        self.cache, self.entries = cache, entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self,i):
        row = self.entries[i]
        label = row['reference_class'] if row['reference_class'] is not None else -1
        return torch.from_numpy(self.cache[row['input_index']].astype(np.float32)),label

class SpotCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2,12,3,padding=1),nn.GroupNorm(3,12),nn.SiLU())
        self.fine = nn.Sequential(nn.Conv2d(12,12,3,padding=1),nn.GroupNorm(3,12),nn.SiLU())
        self.coarse = nn.Sequential(nn.Conv2d(12,12,5,padding=2),nn.GroupNorm(3,12),nn.SiLU())
        self.mix = nn.Sequential(nn.Conv2d(24,24,3,padding=1),nn.GroupNorm(6,24),nn.SiLU(),nn.AvgPool2d(2),
                                 nn.Conv2d(24,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU())
        self.head = nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Dropout(.2),nn.Linear(32,2))

    def forward(self,x):
        valid = x[:,2:3]
        h = F.avg_pool2d(self.stem(x[:,:2]),2)
        h = self.mix(torch.cat((self.fine(h),self.coarse(h)),dim=1))
        weights = F.avg_pool2d(valid,4)
        avg = (h*weights).sum(dim=(2,3))/weights.sum(dim=(2,3)).clamp_min(1)
        maximum = h.masked_fill(weights<=.5,torch.finfo(h.dtype).min).amax(dim=(2,3))
        return self.head(torch.cat((avg,maximum),dim=1))


def load_manifest(manifest):
    rows = []
    seen = set()
    experiment_split = {}
    for row in json.loads(manifest.read_text(encoding='utf-8')):
        split = row['split']
        if row.get('duplicate_of') or split in ('excluded', 'M20_challenge'):
            continue
        if split not in ('train', 'historical_test', 'recent'):
            raise ValueError(f'Unknown split {split!r}')
        key = row['member']
        if key in seen:
            raise ValueError(f'Repeated manifest member {key!r}')
        seen.add(key)
        experiment = row['experiment']
        previous_split = experiment_split.setdefault(experiment, split)
        if previous_split != split:
            raise ValueError(f'Experiment {experiment!r} spans {previous_split!r} and {split!r}')
        label = row.get('reference_class')
        if split != 'recent' and label not in (0, 1):
            raise ValueError(f'Missing binary label for {key!r}')
        pitch = float(row['um_per_pixel'])
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError(f'Invalid pixel size for {key!r}')
        rows.append(dict(row, um_per_pixel=pitch, input_index=len(rows)))
    for split in ('train', 'historical_test'):
        labels = {r['reference_class'] for r in rows if r['split'] == split}
        if labels != {0, 1}:
            raise ValueError(f'{split} must contain both classes')
    return rows


def read_tiff(row, root):
    if row.get('archive'):
        with zipfile.ZipFile(root / row['archive']) as archive:
            raw = archive.read(row['member'])
    else:
        raw = (root / row['member']).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        bits = image.tag_v2.get(258)
        if image.mode != 'RGB' or image.n_frames != 1:
            raise ValueError(f'Expected one RGB TIFF plane in {row["member"]}')
        if bits == (16, 16, 16):
            green = decode_12bit_tiff(raw)[:, :, 1].astype(np.float32) * (255.0/4095.0)
            reference = resize(green, 224, 224)
        elif bits == (8, 8, 8):
            green = np.asarray(image.getchannel('G'), dtype=np.float32)
            # Preserve 8-bit rounding in the resized reference used in the analysis.
            reference = np.asarray(
                image.resize((224, 224), Image.Resampling.BILINEAR).getchannel('G'),
                dtype=np.float32,
            )
        else:
            raise ValueError(f'Unsupported TIFF sample depth {bits} in {row["member"]}')
    return green, reference, hashlib.sha256(raw).hexdigest()


def prepare(rows, root, output, view):
    height = FULL_HEIGHT if view == 'full' else CROP_HEIGHT
    cache = np.lib.format.open_memmap(
        output / 'inputs.npy', mode='w+', dtype=np.float16, shape=(len(rows), 3, height, WIDTH),
    )

    def extract(row):
        green, reference, digest = read_tiff(row, root)
        row = dict(row, green_offset=background_level(reference),
                   bright_green_p999=float(np.percentile(reference, 99.9)))
        if view == 'full':
            sampled, valid = sample_full(green, row['green_offset'])
        else:
            sampled, valid = sample_crop(green, row)
        return channels(sampled, valid, row), digest

    seen = set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (pixels, digest) in enumerate(pool.map(extract, rows)):
            if digest in seen:
                raise ValueError(f'Duplicate TIFF content at {rows[index]["member"]!r}')
            seen.add(digest)
            cache[index] = pixels
    cache.flush()
    return cache


def score(model, loader):
    model.eval()
    values = []
    with torch.inference_mode():
        for pixels, _ in loader:
            first = model(pixels).softmax(1)[:, 1]
            mirror = model(pixels.flip(-1)).softmax(1)[:, 1]
            values.extend(((first + mirror) / 2).tolist())
    return np.asarray(values)


def summarize(y, scores):
    calls = scores > .5
    tn, fp, fn, tp = map(int, confusion_matrix(y, calls, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return dict(n=len(y), accuracy=(tn + tp) / len(y), sensitivity=sensitivity,
                specificity=specificity, balanced_accuracy=(sensitivity + specificity) / 2,
                auc=float(roc_auc_score(y, scores)), binary_auc=float(roc_auc_score(y, calls)),
                confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp))


def train(rows, cache, output, view):
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    train_rows = [r for r in rows if r['split'] == 'train']
    evaluation = [r for r in rows if r['split'] != 'train']
    model = SpotCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    train_loader = DataLoader(Inputs(cache, train_rows), batch_size=24, shuffle=True, num_workers=0)
    test_loader = DataLoader(Inputs(cache, evaluation), batch_size=24, shuffle=False, num_workers=0)
    weights = torch.tensor([len(train_rows) / (2 * sum(r['reference_class'] == y for r in train_rows))
                            for y in (0, 1)])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for epoch in range(10):
        started = time.monotonic()
        model.train()
        total = 0.0
        for pixels, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(pixels), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'{view} epoch {epoch + 1}/10, loss {total / len(train_loader):.4f}, '
              f'{time.monotonic() - started:.1f}s', flush=True)

    scores = score(model, test_loader)
    if len(scores) != len(evaluation) or not np.isfinite(scores).all():
        raise RuntimeError('Invalid evaluation scores')
    historical = [i for i, r in enumerate(evaluation) if r['split'] == 'historical_test']
    matched = [i for i in historical if evaluation[i]['um_per_pixel'] < .9]
    summary = summarize([evaluation[i]['reference_class'] for i in historical], scores[historical])
    if {evaluation[i]['reference_class'] for i in matched} == {0, 1}:
        summary['matched_20x'] = summarize(
            [evaluation[i]['reference_class'] for i in matched], scores[matched],
        )
    predictions = []
    for r, value in zip(evaluation, scores):
        label = r.get('reference_class') if r['split'] == 'historical_test' else None
        if r['split'] == 'recent' and r.get('mestranol_uM') == 0:
            label = 0
        predictions.append(dict(member=r['member'], split=r['split'],
                                reference_class=label, mestranol_uM=r.get('mestranol_uM'),
                                class_1_score=float(value), predicted_class=int(value > .5)))
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    torch.save(dict(state_dict=model.state_dict(), variant=view, seed=42, epochs=10,
                    train_members=[r['member'] for r in train_rows],
                    test_members=[r['member'] for r in evaluation]), output / 'model.pth')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path, help='Ordered JSON image manifest')
    parser.add_argument('--root', required=True, type=Path, help='Root containing TIFFs or ZIPs')
    parser.add_argument('--output', required=True, type=Path, help='New directory for outputs')
    parser.add_argument('--view', choices=('full', 'crop'), default='full')
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    cache = prepare(rows, args.root, args.output, args.view)
    train(rows, cache, args.output, args.view)


if __name__ == '__main__':
    main()
