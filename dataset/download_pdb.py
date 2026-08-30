import os
import csv
import requests
from typing import List

INPUT_CSV = "protein_seq.csv"
OUT_DIR = "protein_pdb"
TIMEOUT = 60

os.makedirs(OUT_DIR, exist_ok=True)


def read_uniprot_ids(csv_path: str) -> List[str]:
    ids = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            uid = row[0].strip()
            if uid:
                ids.append(uid)
    return ids


def get_alphafold_pdb_url(uniprot_id: str):
    """
    Query AlphaFold DB API.
    Return pdbUrl if exists, else None.
    """
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    r = requests.get(api, timeout=TIMEOUT)
    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    # API 返回通常是 list
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("pdbUrl", None)
    return None


def download_pdb(url: str, out_path: str):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def main():
    uniprot_ids = read_uniprot_ids(INPUT_CSV)
    print(f"Found {len(uniprot_ids)} UniProt IDs")

    failed_ids = []

    for idx, uid in enumerate(uniprot_ids, 1):
        print(f"[{idx}/{len(uniprot_ids)}] Processing {uid}")

        try:
            pdb_url = get_alphafold_pdb_url(uid)
            if pdb_url is None:
                print(f"  ✗ No PDB found for {uid}")
                failed_ids.append(uid)
                continue

            out_pdb = os.path.join(OUT_DIR, f"{uid}.pdb")

            # 已存在就跳过（可按需删掉）
            if os.path.exists(out_pdb):
                print(f"  ✓ Exists: {out_pdb}")
                continue

            download_pdb(pdb_url, out_pdb)
            print(f"  ✓ Downloaded: {out_pdb}")

        except Exception as e:
            print(f"  ✗ Error for {uid}: {e}")
            failed_ids.append(uid)

    print("\n==== UniProt IDs WITHOUT PDB ====")
    for uid in failed_ids:
        print(uid)

    print(f"\nTotal failed: {len(failed_ids)}")


if __name__ == "__main__":
    main()
