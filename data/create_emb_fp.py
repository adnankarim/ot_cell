import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def mol_to_morgan_bits(smiles: str, radius: int = 2, n_bits: int = 1024) -> list[int] | None:
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return list(fp)


def main():
    metadata_csv = sys.argv[1] if len(sys.argv) > 1 else "/home/jovyan/bbbc021_all/metadata/bbbc021_df_all.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "/home/jovyan/emb_fp.csv"

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    name_col = find_col(df, ["CPD_NAME", "Compound", "compound", "compound_name", "drug", "perturbation"])
    smiles_col = find_col(df, ["SMILES", "smiles"])

    if name_col is None:
        raise ValueError(
            f"Could not find compound-name column. Available columns:\n{list(df.columns)}"
        )
    if smiles_col is None:
        raise ValueError(
            f"Could not find SMILES column. Available columns:\n{list(df.columns)}"
        )

    pairs = df[[name_col, smiles_col]].dropna().drop_duplicates()

    rows = {}
    failed = []

    for _, row in pairs.iterrows():
        compound_name = str(row[name_col]).strip()
        smiles = str(row[smiles_col]).strip()

        bits = mol_to_morgan_bits(smiles, radius=2, n_bits=1024)
        if bits is None:
            failed.append((compound_name, smiles))
            continue

        rows[compound_name] = bits

    if not rows:
        raise RuntimeError("No valid embeddings were generated. Check your SMILES column.")

    emb_df = pd.DataFrame.from_dict(rows, orient="index")
    emb_df.index.name = "compound"

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    emb_df.to_csv(output_csv)

    print(f"Saved embedding file to: {output_csv}")
    print(f"Valid compounds: {len(rows)}")
    print(f"Failed compounds: {len(failed)}")

    if failed:
        failed_csv = os.path.splitext(output_csv)[0] + "_failed.csv"
        pd.DataFrame(failed, columns=["compound", "smiles"]).to_csv(failed_csv, index=False)
        print(f"Failed rows saved to: {failed_csv}")


if __name__ == "__main__":
    main()