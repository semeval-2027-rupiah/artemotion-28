"""Compare ST1 human captions, aligned visual captions and wrong-image captions."""

import argparse
import random
import sys

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline


EMOTIONS = {'amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness'}
VISUAL = 'visual_caption_en'
COLUMNS = ['split', 'image_name', 'language', 'emotion', 'caption', VISUAL]


def classifier(seed):
    return make_pipeline(
        HashingVectorizer(analyzer='char', ngram_range=(2, 4), n_features=2**19),
        SGDClassifier(loss='hinge', random_state=seed),
    )


def swapped_captions(val, seed):
    # Rotate a shuffled list of distinct images, not rows: repeated annotations
    # of a painting must never supply its own negative control.
    captions = val.drop_duplicates('image_name').set_index('image_name')[VISUAL]
    images = sorted(captions.index)
    if len(images) < 2:
        raise ValueError('Swapped control requires at least two validation images.')
    random.Random(seed).shuffle(images)
    donors = dict(zip(images, images[1:] + images[:1]))
    return val.image_name.map(donors).map(captions)


def evaluate(frame, seed):
    rows = frame.loc[frame.split.isin(['train', 'val']) & frame.emotion.isin(EMOTIONS)].copy()
    for column in COLUMNS:
        if rows[column].str.strip().eq('').any():
            raise ValueError(f'Blank {column} on eligible rows; supply a complete evaluation CSV.')
    if rows.groupby('image_name')[VISUAL].nunique().gt(1).any():
        raise ValueError('Expected one visual caption per image_name.')
    train = rows.loc[rows.split == 'train']
    val = rows.loc[rows.split == 'val'].copy()
    if train.emotion.nunique() < 2:
        raise ValueError('Need at least two ST1 emotions in train.')
    swapped = swapped_captions(val, seed)
    text = classifier(seed).fit(train.caption, train.emotion)
    val['text_only'] = text.predict(val.caption)
    fusion = classifier(seed).fit(train.caption + '\n' + train[VISUAL], train.emotion)
    val['aligned'] = fusion.predict(val.caption + '\n' + val[VISUAL])
    val['swapped'] = fusion.predict(val.caption + '\n' + swapped)

    results = []
    for language, group in val.groupby('language'):
        scores = {col: f1_score(group.emotion, group[col], average='macro', zero_division=0)
                  for col in ('text_only', 'aligned', 'swapped')}
        results.append(dict(language=language, n_val=len(group), **scores))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='Enriched train/val CSV with columns: ' + ', '.join(COLUMNS))
    parser.add_argument('--seed', type=int, default=42, help='Classifier and swap seed (default: 42)')
    args = parser.parse_args()
    try:
        rows = pd.read_csv(args.csv, usecols=COLUMNS, dtype=str, keep_default_na=False)
        evaluate(rows, args.seed).to_csv(sys.stdout, index=False, float_format='%.6f')
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
