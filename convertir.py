import os
import glob
import numpy as np
import pandas as pd
from matio import load_from_mat

def convert(mat_path, csv_path):
    data = load_from_mat(mat_path, raw_data=False, add_table_attrs=True)
    table = list(data.values())[0]
    first_col = table.iloc[:, 0]
    is_signal = isinstance(first_col.iloc[0], np.ndarray)

    if is_signal:
        channels = table.columns.tolist()
        fs = len(np.asarray(table[channels[0]].iloc[0]).flatten())
        n_samples = len(table) * fs
        flat = {ch: np.concatenate([np.asarray(v).flatten() for v in table[ch].values])
                for ch in channels}
        df = pd.DataFrame(flat)
        df.insert(0, "Time", np.arange(n_samples) / fs)
        kind = "signal EEG"
    else:
        df = table.reset_index()
        df["Onset"] = df["Onset"].dt.total_seconds()
        kind = "journal d'annotations"

    df.to_csv(csv_path, index=False)
    print(f"[{kind}] {csv_path} -> {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Dossier source
dossier_source = r"C:\Users\LENOVO\Desktop\Projet_EEG_DataMining\Projet data mining 3AINFO"
dossier_destination = os.path.join(dossier_source, "data_csv")
os.makedirs(dossier_destination, exist_ok=True)

# Trouver tous les .mat
mat_files = sorted(glob.glob(os.path.join(dossier_source, "*.mat")))
print(f"Trouvé {len(mat_files)} fichier(s) .mat\n")

for mat_path in mat_files:
    base = os.path.splitext(os.path.basename(mat_path))[0]
    csv_path = os.path.join(dossier_destination, f"{base}.csv")
    try:
        convert(mat_path, csv_path)
    except Exception as e:
        print(f"Erreur sur {mat_path}: {e}")