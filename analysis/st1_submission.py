"""Validate an ST1 Codabench ZIP against the IDs in a phase input file."""

import argparse
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from common import EMOTIONS, PHASE_INPUTS

DEV_INPUTS = PHASE_INPUTS / "dev" / "st1_test.csv"
SUBMISSION_FILES = ("predictions_text.csv", "predictions_multimodal.csv")


def validate_submission(archive_path, input_path):
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError("Phase inputs have no id column")
        ids = [row["id"] for row in reader]
    expected = set(ids)
    if not expected or not all(expected) or len(expected) != len(ids):
        raise ValueError("Phase inputs must have nonempty, unique IDs")

    with ZipFile(archive_path) as archive:
        if sorted(archive.namelist()) != sorted(SUBMISSION_FILES):
            raise ValueError("ZIP must contain exactly the two prediction CSVs at the root")
        for name in SUBMISSION_FILES:
            seen = set()
            with archive.open(name) as member, io.TextIOWrapper(member, encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                if next(reader, None) != ["id", "label"]:
                    raise ValueError(f"{name}: expected id,label header")
                for row in reader:
                    if len(row) != 2 or row[1] not in EMOTIONS:
                        raise ValueError(f"{name}:{reader.line_num}: expected an ID and a valid lowercase label")
                    row_id = row[0]
                    if row_id not in expected or row_id in seen:
                        raise ValueError(f"{name}:{reader.line_num}: unknown or duplicate ID {row_id!r}")
                    seen.add(row_id)
            if missing := expected - seen:
                raise ValueError(f"{name}: missing {len(missing)} IDs (first: {min(missing)!r})")
    return len(expected)


def write_submission(path, input_path, ids, text_labels, multimodal_labels):
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        with ZipFile(staged, "w") as archive:
            for name, labels in zip(SUBMISSION_FILES, (text_labels, multimodal_labels), strict=True):
                content = io.StringIO(newline="")
                writer = csv.writer(content, lineterminator="\n")
                writer.writerow(["id", "label"])
                writer.writerows((row_id, EMOTIONS[label]) for row_id, label in zip(ids, labels, strict=True))
                archive.writestr(name, content.getvalue())
        count = validate_submission(staged, input_path)
        staged.replace(path)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inputs", type=Path, default=DEV_INPUTS)
    args = parser.parse_args()
    try:
        count = validate_submission(args.archive, args.inputs)
    except (BadZipFile, csv.Error, OSError, ValueError) as error:
        parser.exit(1, f"Invalid ST1 submission: {error}\n")
    print(f"{args.archive}: valid format, {count:,} IDs in each CSV")


if __name__ == "__main__":
    main()
