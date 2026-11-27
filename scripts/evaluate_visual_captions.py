"""Compare ST1 human captions, aligned visual captions and wrong-image captions."""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from common import EMOTION_INDEX, normalise_language  # noqa: E402
from metrics import macro_f1  # noqa: E402

VISUAL = 'visual_caption_en'
COLUMNS = ['split', 'image_name', 'language', 'emotion', 'caption', VISUAL]
PREDICTIONS = {'text_only': 'pred_text', 'aligned': 'pred_aligned', 'swapped': 'pred_swapped'}


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


def evaluate(frame, seed, *, predictions_out=None):
    columns = COLUMNS if predictions_out is None else [*COLUMNS, 'painting']
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing columns: {", ".join(missing)}')
    rows = frame.loc[frame.split.isin(['train', 'val']) & frame.emotion.isin(EMOTION_INDEX)].copy()
    for column in columns:
        if rows[column].str.strip().eq('').any():
            raise ValueError(f'Blank {column} on eligible rows; supply a complete evaluation CSV.')
    if rows.groupby('image_name')[VISUAL].nunique().gt(1).any():
        raise ValueError('Expected one visual caption per image_name.')
    rows['label'] = rows.emotion.map(EMOTION_INDEX).astype(int)
    rows['language'] = normalise_language(rows['language']).to_numpy()
    train = rows.loc[rows.split == 'train']
    val = rows.loc[rows.split == 'val'].copy()
    if train.label.nunique() < 2:
        raise ValueError('Need at least two ST1 emotions in train.')
    swapped = swapped_captions(val, seed)
    text = classifier(seed).fit(train.caption, train.label)
    val['pred_text'] = text.predict(val.caption)
    fusion = classifier(seed).fit(train.caption + '\n' + train[VISUAL], train.label)
    val['pred_aligned'] = fusion.predict(val.caption + '\n' + val[VISUAL])
    val['pred_swapped'] = fusion.predict(val.caption + '\n' + swapped)

    if predictions_out is not None:
        val[['painting', 'language', 'label', *PREDICTIONS.values()]].to_csv(predictions_out, index=False)

    results = []
    for language, group in val.groupby('language'):
        scores = {name: macro_f1(group.label, group[column]) for name, column in PREDICTIONS.items()}
        results.append(dict(language=language, n_val=len(group), **scores))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, help='Enriched train/val CSV with columns: ' + ', '.join(COLUMNS))
    parser.add_argument('--seed', type=int, default=42, help='Classifier and swap seed (default: 42)')
    parser.add_argument('--predictions-out', type=Path,
                        help='Write aligned val predictions for analysis/score.py (requires painting in input)')
    args = parser.parse_args()
    if args.predictions_out is not None and args.predictions_out.resolve() == args.csv.resolve():
        parser.error('--predictions-out must differ from the input CSV')
    try:
        rows = pd.read_csv(args.csv, usecols=lambda col: col in COLUMNS or col == 'painting',
                           dtype=str, keep_default_na=False)
        report = evaluate(rows, args.seed, predictions_out=args.predictions_out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report.to_csv(sys.stdout, index=False, float_format='%.6f')
    headline = ', '.join(f'{name}={report[name].mean():.6f}' for name in PREDICTIONS)
    print(f'ST1 language-macro F1 ({report.n_val.sum()} validation rows): {headline}', file=sys.stderr)


if __name__ == '__main__':
    main()
