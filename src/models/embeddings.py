import numpy as np, pickle, json, time
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR/"data"/"processed"
MODELS_DIR     = BASE_DIR/"models_saved"
MODELS_DIR.mkdir(exist_ok=True)

def gerar_tfidf(splits):
    print("\n  Gerando TF-IDF + SVD...")
    t0 = time.time()
    vec = TfidfVectorizer(max_features=20000,ngram_range=(1,2),min_df=3,
                          max_df=0.95,sublinear_tf=True,
                          token_pattern=r'\b[a-z]{3,}\b')
    X_tr = vec.fit_transform(splits["X_train"])
    X_v  = vec.transform(splits["X_val"])
    X_te = vec.transform(splits["X_test"])
    print(f"    Vocabulario: {len(vec.vocabulary_):,} termos")
    svd = TruncatedSVD(n_components=300,random_state=42)
    X_tr = svd.fit_transform(X_tr)
    X_v  = svd.transform(X_v)
    X_te = svd.transform(X_te)
    var = svd.explained_variance_ratio_.sum()
    print(f"    Variancia SVD: {var:.1%} | Shape: {X_tr.shape}")
    with open(MODELS_DIR/"tfidf_vectorizer.pkl","wb") as f: pickle.dump(vec,f)
    with open(MODELS_DIR/"svd_lsa.pkl","wb") as f: pickle.dump(svd,f)
    np.save(MODELS_DIR/"tfidf_X_train.npy",X_tr)
    np.save(MODELS_DIR/"tfidf_X_val.npy",X_v)
    np.save(MODELS_DIR/"tfidf_X_test.npy",X_te)
    print(f"    OK em {time.time()-t0:.1f}s")
    return {"X_train":X_tr,"X_val":X_v,"X_test":X_te}

def gerar_sbert(splits, batch_size=32):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers nao instalado. Pulando SBERT.")
        return {}
    print("\n  Gerando Sentence Embeddings (SBERT)...")
    print("  Modelo: paraphrase-multilingual-MiniLM-L12-v2")
    print("  (1a execucao: download ~130MB...)")
    t0 = time.time()
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    def enc(textos, desc):
        return model.encode(textos,batch_size=batch_size,
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            normalize_embeddings=True)
    X_tr = enc(splits["X_train"].tolist(),"treino")
    X_v  = enc(splits["X_val"].tolist(),"val")
    X_te = enc(splits["X_test"].tolist(),"teste")
    np.save(MODELS_DIR/"sbert_X_train.npy",X_tr)
    np.save(MODELS_DIR/"sbert_X_val.npy",X_v)
    np.save(MODELS_DIR/"sbert_X_test.npy",X_te)
    print(f"    OK em {time.time()-t0:.1f}s | Shape: {X_tr.shape}")
    return {"X_train":X_tr,"X_val":X_v,"X_test":X_te}

def gerar_bert(splits, batch_size=8, max_length=512):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("  transformers/torch nao instalado. Pulando BERTimbau.")
        return {}
    print("\n  Gerando BERTimbau Embeddings...")
    print("  Modelo: neuralmind/bert-base-portuguese-cased")
    print("  AVISO: 1a execucao baixa ~1.3GB. Pode demorar.")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Dispositivo: {device}")
    tok = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = model.to(device).eval()

    def mean_pool(out, mask):
        emb = out[0]
        mask_exp = mask.unsqueeze(-1).expand(emb.size()).float()
        return torch.sum(emb*mask_exp,1) / torch.clamp(mask_exp.sum(1),min=1e-9)

    def encode(textos, desc):
        all_emb = []
        for i in tqdm(range(0,len(textos),batch_size),desc=f"  {desc}"):
            batch = textos[i:i+batch_size]
            enc = tok(batch,padding=True,truncation=True,
                      max_length=max_length,return_tensors="pt")
            enc = {k:v.to(device) for k,v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            emb = mean_pool(out,enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb,p=2,dim=1)
            all_emb.append(emb.cpu().numpy())
        return np.vstack(all_emb)

    X_tr = encode(splits["X_train"].tolist(),"treino BERT")
    X_v  = encode(splits["X_val"].tolist(),"val BERT")
    X_te = encode(splits["X_test"].tolist(),"teste BERT")
    np.save(MODELS_DIR/"bert_X_train.npy",X_tr)
    np.save(MODELS_DIR/"bert_X_val.npy",X_v)
    np.save(MODELS_DIR/"bert_X_test.npy",X_te)
    print(f"    OK em {(time.time()-t0)/60:.1f}min | Shape: {X_tr.shape}")
    return {"X_train":X_tr,"X_val":X_v,"X_test":X_te}

def executar(usar_bert=True):
    print("\n"+"="*60+"\n  ETAPA 4 - EMBEDDINGS\n"+"="*60)
    with open(DATA_PROCESSED/"splits.pkl","rb") as f: splits = pickle.load(f)
    print(f"\n  Treino={len(splits['X_train'])} | Val={len(splits['X_val'])} | Teste={len(splits['X_test'])}")
    gerar_tfidf(splits)
    gerar_sbert(splits)
    if usar_bert:
        try: gerar_bert(splits)
        except Exception as e: print(f"\n  BERTimbau falhou: {e}\n  Continuando sem ele.")
    else:
        print("\n  BERTimbau pulado (--no-bert)")
    print(f"\n  Embeddings salvos em: {MODELS_DIR}")
    print("\nPROXIMA ETAPA: python src/models/treinamento.py\n")

if __name__ == "__main__":
    import sys
    executar(usar_bert="--no-bert" not in sys.argv)
