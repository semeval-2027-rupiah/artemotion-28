# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["transformers==4.46.3", "torch==2.5.1", "torchvision==0.20.1", "accelerate>=0.26,<2", "einops>=0.7", "pillow"]
# ///
"""Add an image-only English description to each selected annotation row."""

import argparse
import csv
import hashlib
import os
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile

import caption_models


def generate(annotations, images, out_dir, backend, limit=None):
    images = images.resolve()
    dest = out_dir / (f'sample-{limit}.csv' if limit else 'augmented.csv')
    if annotations.resolve() == dest.resolve():
        raise ValueError('Output would overwrite the input CSV')
    with annotations.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if 'image_name' not in fields or 'visual_caption_en' in fields:
            raise ValueError('Expected image_name and no existing visual_caption_en')
        names = {row['image_name'] for row in reader}
    if not names or not all(names):
        raise ValueError('Missing image names or empty input CSV')
    # Stable, style-independent sample shared by both backends.
    selected = sorted(names, key=lambda n: hashlib.sha256(n.encode()).digest())[:limit]
    for name in selected:
        path = (images / name).resolve()
        if not path.is_relative_to(images) or not path.is_file():
            raise FileNotFoundError(f'Invalid or missing image: {path}')

    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(caption_models.__file__).read_bytes()).hexdigest()
    key = f'{backend}:{images}:{digest}'
    captions = {}
    captioner = None
    db = sqlite3.connect(out_dir / 'captions.sqlite')
    try:
        db.execute('CREATE TABLE IF NOT EXISTS cache (config TEXT, name TEXT, caption TEXT, PRIMARY KEY(config,name))')
        for i, name in enumerate(selected, 1):
            hit = db.execute('SELECT caption FROM cache WHERE config=? AND name=?', (key, name)).fetchone()
            if hit:
                captions[name] = hit[0]
            else:
                if captioner is None:
                    captioner = caption_models.load_captioner(backend)
                try:
                    text = captioner(images / name).strip()
                    if not text:
                        raise ValueError('Empty caption')
                except Exception as exc:
                    raise RuntimeError(f'Captioning failed for {name}') from exc
                with db:
                    db.execute('INSERT INTO cache VALUES (?,?,?)', (key, name, text))
                captions[name] = text
            if i == 1 or i % 25 == 0 or i == len(selected):
                print(f'{i}/{len(selected)} images ready', flush=True)
    finally:
        db.close()

    rows = 0
    with NamedTemporaryFile(mode='w', dir=out_dir, encoding='utf-8', newline='', delete=False) as tmp:
        try:
            writer = csv.DictWriter(tmp, fieldnames=[*fields, 'visual_caption_en'])
            writer.writeheader()
            with annotations.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    if row['image_name'] in captions:
                        writer.writerow(dict(row, visual_caption_en=captions[row['image_name']]))
                        rows += 1
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise
    os.replace(tmp.name, dest)
    print(f'{rows} annotation rows, {len(captions)} paintings -> {dest}')
    return dest


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['molmo', 'blip'], default='molmo')
    parser.add_argument('--annotations', type=Path, default=root / 'annotations/artelingo28_train_val.csv')
    parser.add_argument('--images', type=Path, default=root / 'images')
    parser.add_argument('--out-dir', type=Path, help='Default: runs/visual-captions/<backend>')
    parser.add_argument('--limit', type=int, help='Pilot: include only this many unique paintings')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    generate(args.annotations, args.images, args.out_dir or root / 'runs/visual-captions' / args.backend,
             args.backend, args.limit)
