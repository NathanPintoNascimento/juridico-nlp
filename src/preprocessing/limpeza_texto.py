import re, unicodedata, pandas as pd, numpy as np, pickle, json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
DATA_RAW       = BASE_DIR/"data"/"raw"
DATA_PROCESSED = BASE_DIR/"data"/"processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

STOPWORDS = {"a","ao","aos","as","com","da","das","de","do","dos","e","em","na","nas",
             "no","nos","o","os","ou","para","pela","pelas","pelo","pelos","por","que",
             "se","sem","seu","seus","sua","suas","um","uma","uns","umas","acordao",
             "relator","ministro","camara","turma","provido","desprovido","negado",
             "concedido","mantido","autos","presentes","parte","partes"}

_RE_PROC   = re.compile(r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}')
_RE_NUMS   = re.compile(r'\b\d+\b')
_RE_PUNCT  = re.compile(r'[^\w\s]')
_RE_SPACES = re.compile(r'\s+')

class LimpadorJuridico:
    def __init__(self, modo="classico", remover_stopwords=True):
        self.modo = modo
        self.remover_stopwords = remover_stopwords

    def limpar(self, texto):
        if not isinstance(texto,str) or not texto.strip(): return ""
        texto = unicodedata.normalize("NFC", texto)
        texto = _RE_PROC.sub(" ", texto)
        if self.modo == "classico":
            texto = texto.lower()
            texto = _RE_PUNCT.sub(" ", texto)
            texto = _RE_NUMS.sub(" ", texto)
            if self.remover_stopwords:
                texto = " ".join(p for p in texto.split()
                                 if p not in STOPWORDS and len(p)>2)
        return _RE_SPACES.sub(" ", texto).strip()

    def limpar_batch(self, textos, desc="Limpando"):
        return [self.limpar(t) for t in tqdm(textos, desc=f"  {desc}")]

def criar_splits(df, test_size=0.2, val_size=0.1):
    df = df[df["texto_limpo"].str.len()>10].copy()
    X,y = df["texto_limpo"].values, df["classe"].values
    X_tv,X_te,y_tv,y_te = train_test_split(
        X,y,test_size=test_size,stratify=y,random_state=42)
    X_tr,X_v,y_tr,y_v = train_test_split(
        X_tv,y_tv,test_size=val_size/(1-test_size),stratify=y_tv,random_state=42)
    print(f"\n  Treino={len(X_tr)} | Val={len(X_v)} | Teste={len(X_te)}")
    return {"X_train":X_tr,"y_train":y_tr,"X_val":X_v,
            "y_val":y_v,"X_test":X_te,"y_test":y_te}

def executar():
    print("\n"+"="*60+"\n  ETAPA 3 - PRE-PROCESSAMENTO\n"+"="*60)
    p = DATA_RAW/"dataset_juridico.parquet"
    if not p.exists(): p = DATA_RAW/"dataset_juridico.csv"
    df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    logger.info(f"Dataset: {df.shape}")

    lc = LimpadorJuridico(modo="classico")
    ln = LimpadorJuridico(modo="neural", remover_stopwords=False)
    df["texto_limpo"]  = lc.limpar_batch(df["texto"].tolist(), "Classico")
    df["texto_neural"] = ln.limpar_batch(df["texto"].tolist(), "Neural")

    le = LabelEncoder()
    df["classe_id"] = le.fit_transform(df["classe"])
    md = BASE_DIR/"models_saved"; md.mkdir(exist_ok=True)
    with open(md/"label_encoder.pkl","wb") as f: pickle.dump(le,f)

    print("\n  Classes:")
    for i,c in enumerate(le.classes_): print(f"    {i} -> {c}")

    splits = criar_splits(df)
    df.to_parquet(DATA_PROCESSED/"dataset_processado.parquet",index=False)
    with open(DATA_PROCESSED/"splits.pkl","wb") as f: pickle.dump(splits,f)
    with open(DATA_PROCESSED/"metadata.json","w") as f:
        json.dump({"classes":le.classes_.tolist(),
                   "n_classes":len(le.classes_),
                   "n_train":len(splits["X_train"]),
                   "n_val":len(splits["X_val"]),
                   "n_test":len(splits["X_test"])},f,indent=2)
    print(f"\n  Salvo em: {DATA_PROCESSED}")
    print("\nPROXIMA ETAPA: python src/models/embeddings.py\n")
    return df, splits, le

if __name__ == "__main__":
    import os; os.makedirs("logs",exist_ok=True)
    executar()
