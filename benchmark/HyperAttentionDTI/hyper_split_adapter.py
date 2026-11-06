"""Adapt DCGCN split CSVs into the exact five-field HyperAttentionDTI sample strings.

No model or tokenizer changes. HyperAttentionDTI's existing collate_fn expects:
   drug_id protein_id SMILES amino_acid_sequence label
(whitespace separated; it uses the final five tokens).
"""
from pathlib import Path
import csv


def _lookup(path):
    result = {}
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        for i, row in enumerate(csv.reader(stream), start=1):
            if not row or not any(col.strip() for col in row):
                continue
            if len(row) != 2:
                raise ValueError(f'{path}:{i}: expected ID,sequence (two columns, no header)')
            identity, sequence = (x.strip() for x in row)
            if not identity or not sequence:
                raise ValueError(f'{path}:{i}: ID/sequence is empty')
            if identity in result and result[identity] != sequence:
                raise ValueError(f'{path}:{i}: ID {identity!r} maps to different sequences')
            if any(c.isspace() for c in identity) or any(c.isspace() for c in sequence):
                raise ValueError(f'{path}:{i}: whitespace not allowed in ID/sequence')
            result[identity] = sequence
    if not result:
        raise ValueError(f'No usable records in {path}')
    return result


def _protein_file(root):
    for name in ('protein_seq.csv', 'prorein_seq.csv', 'protein_seq.txt'):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'Cannot find protein_seq.csv/prorein_seq.csv/protein_seq.txt in {root}')


def _triples(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing split CSV: {csv_path}')
    rows = []
    with csv_path.open('r', newline='', encoding='utf-8-sig') as stream:
        for i, row in enumerate(csv.reader(stream), start=1):
            if not row or not any(x.strip() for x in row):
                continue
            if len(row) != 3:
                raise ValueError(f'{csv_path}:{i}: expected drug_id,protein_id,0/1')
            did, pid, y = [x.strip() for x in row]
            if not did or not pid or y not in ('0', '1'):
                raise ValueError(f'{csv_path}:{i}: invalid pair or label: {row}')
            rows.append((did, pid, y))
    if not rows:
        raise ValueError(f'No pairs in {csv_path}')
    if len(set(rows)) != len(rows):
        raise ValueError(f'Duplicate rows in {csv_path}')
    if len({(d, p) for d, p, _ in rows}) != len(rows):
        raise ValueError(f'A DTI pair has conflicting/duplicate labels in {csv_path}')
    return rows


def load_pair_splits(dataset_root, task, split, fold):
    """Return train/test lists for CustomDataSet with no additions or dropped pairs."""
    root = Path(dataset_root).expanduser().resolve() / task
    drug_map = _lookup(root / 'drug_SMILES.csv')
    protein_map = _lookup(_protein_file(root))
    split_dir = root / 'result' / split
    trains = _triples(split_dir / f'train_cv{fold}.csv')
    tests = _triples(split_dir / f'test_cv{fold}.csv')

    overlap = {(d, p) for d, p, _ in trains} & {(d, p) for d, p, _ in tests}
    if overlap:
        raise ValueError(f'Same DTI pair occurs in train and test: {sorted(overlap)[:5]}')

    # Require both labels in external testing split for ROC/AUC.
    if {y for _, _, y in tests} != {'0', '1'}:
        raise ValueError(f'Test split must have positive and negative pairs: {split_dir}')

    # Independently verify entity groups if splitter saved its mapping file.
    group_csv = split_dir / 'entity_groups.csv'
    if group_csv.is_file() and split in ('drug_scaffold', 'protein_similarity40'):
        with group_csv.open('r', newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            group_col = 'group_id' if 'group_id' in (reader.fieldnames or []) else 'group'
            if 'entity_id' not in (reader.fieldnames or []) or group_col not in (reader.fieldnames or []):
                raise ValueError(f'Bad entity_groups.csv columns in {group_csv}')
            group_by_id = {row['entity_id']: row[group_col] for row in reader}
        index = 0 if split == 'drug_scaffold' else 1
        try:
            train_groups = {group_by_id[row[index]] for row in trains}
            test_groups = {group_by_id[row[index]] for row in tests}
        except KeyError as e:
            raise ValueError(f'Unknown group member {e} in {group_csv}') from e
        if train_groups & test_groups:
            raise ValueError(f'Group leakage in {split}: {len(train_groups & test_groups)} shared groups')

    def materialize(rows):
        out = []
        for d, p, y in rows:
            if d not in drug_map:
                raise ValueError(f'Drug {d!r} missing from {root / "drug_SMILES.csv"}')
            if p not in protein_map:
                raise ValueError(f'Protein {p!r} missing from sequence lookup')
            out.append(f'{d} {p} {drug_map[d]} {protein_map[p]} {y}')
        return out

    train_samples = materialize(trains)
    test_samples = materialize(tests)
    print(f'[DCGCN split] {root}, {split}, cv{fold}; train={len(trains)} test={len(tests)}', flush=True)
    print(f'[HyperAttentionDTI] Constructed {len(train_samples)} train + {len(test_samples)} test five-field samples', flush=True)
    return train_samples, test_samples
