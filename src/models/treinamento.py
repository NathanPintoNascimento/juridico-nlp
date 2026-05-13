import numpy as np, pandas as pd, pickle, json, time, warnings
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import warnings; warnings.filterwarnings("ignore")

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR/"data"/"processed"
MODELS_DIR     = BASE_DIR/"models_saved"
DOCS_DIR       = BASE_DIR/"docs"
DOCS_DIR.mkdir(exist_ok=True)

def calcular_ks(y_true, y_prob, classes):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=classes)
    ks_scores = []
    for i in range(len(classes)):
        y_t = y_bin[:,i]; y_s = y_prob[:,i]
        idx = np.argsort(y_s)[::-1]
        y_t = y_t[idx]
        n_pos = y_t.sum(); n_neg = len(y_t)-n_pos
        if n_pos==0 or n_neg==0: continue
        cum_pos = np.cumsum(y_t)/n_pos
        cum_neg = np.cumsum(1-y_t)/n_neg
        ks_scores.append(np.max(np.abs(cum_pos-cum_neg)))
    return float(np.mean(ks_scores)) if ks_scores else 0.0

def carregar(tipo="sbert"):
    pref = tipo
    p = MODELS_DIR/f"{pref}_X_train.npy"
    if not p.exists():
        print(f"  Embeddings '{tipo}' nao encontrados. Usando tfidf.")
        pref = "tfidf"
    X_tr = np.load(MODELS_DIR/f"{pref}_X_train.npy")
    X_v  = np.load(MODELS_DIR/f"{pref}_X_val.npy")
    X_te = np.load(MODELS_DIR/f"{pref}_X_test.npy")
    with open(DATA_PROCESSED/"splits.pkl","rb") as f: sp = pickle.load(f)
    with open(MODELS_DIR/"label_encoder.pkl","rb") as f: le = pickle.load(f)
    return X_tr,X_v,X_te,sp["y_train"],sp["y_val"],sp["y_test"],le

def definir_modelos():
    modelos = {
        "logistic_regression": LogisticRegression(
            C=1.0,max_iter=1000,solver="lbfgs",
            class_weight="balanced",random_state=42,n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=200,min_samples_leaf=2,
            class_weight="balanced",random_state=42,n_jobs=-1),
    }
    try:
        import xgboost as xgb
        modelos["xgboost"] = xgb.XGBClassifier(
            n_estimators=300,learning_rate=0.1,max_depth=6,
            subsample=0.8,colsample_bytree=0.8,
            use_label_encoder=False,eval_metric="mlogloss",
            random_state=42,n_jobs=-1)
    except ImportError:
        print("  XGBoost nao instalado. Treinando sem ele.")
    return modelos

def avaliar(modelo, X_te, y_te, classes, tempo):
    y_pred = modelo.predict(X_te)
    y_prob = modelo.predict_proba(X_te)
    acc  = accuracy_score(y_te,y_pred)
    f1   = f1_score(y_te,y_pred,average="macro",zero_division=0)
    try:   auc = roc_auc_score(y_te,y_prob,multi_class="ovr",average="macro")
    except: auc = 0.0
    ks = calcular_ks(y_te,y_prob,classes)
    return {"accuracy":acc,"f1_macro":f1,"auc_macro":auc,"ks":ks,
            "tempo_treino_s":tempo,"cm":confusion_matrix(y_te,y_pred,labels=classes)}

def treinar(tipo="sbert"):
    print("\n"+"="*60+f"\n  ETAPA 5 - TREINAMENTO [{tipo.upper()}]\n"+"="*60)
    X_tr,X_v,X_te,y_tr,y_v,y_te,le = carregar(tipo)
    classes = le.classes_.tolist()
    print(f"\n  Treino: {X_tr.shape} | Classes: {classes}")
    modelos = definir_modelos()
    resultados = {}
    for nome, modelo in modelos.items():
        print(f"\n  Treinando {nome}...")
        y_fit = le.transform(y_tr) if "xgboost" in nome else y_tr
        t0 = time.time()
        modelo.fit(X_tr, y_fit)
        tempo = time.time()-t0
        if "xgboost" in nome:
            y_pred_te = le.inverse_transform(modelo.predict(X_te))
            y_prob_te = modelo.predict_proba(X_te)
            acc  = accuracy_score(y_te,y_pred_te)
            f1   = f1_score(y_te,y_pred_te,average="macro",zero_division=0)
            try:   auc = roc_auc_score(y_te,y_prob_te,multi_class="ovr",average="macro")
            except: auc = 0.0
            ks = calcular_ks(y_te,y_prob_te,list(range(len(classes))))
            m = {"accuracy":acc,"f1_macro":f1,"auc_macro":auc,"ks":ks,
                 "tempo_treino_s":tempo,"cm":confusion_matrix(y_te,y_pred_te,labels=classes)}
        else:
            m = avaliar(modelo, X_te, y_te, classes, tempo)
        resultados[nome] = m
        with open(MODELS_DIR/f"{nome}.pkl","wb") as f: pickle.dump(modelo,f)
        print(f"    AUC={m['auc_macro']:.4f} | F1={m['f1_macro']:.4f} | "
              f"KS={m['ks']:.4f} | Acc={m['accuracy']:.4f} | {tempo:.1f}s")

    print("\n"+"="*60)
    print("  TABELA COMPARATIVA:")
    print(f"  {'Modelo':<22} {'AUC':>6} {'F1':>6} {'KS':>6} {'Acc':>6} {'Tempo':>7}")
    print("  "+"-"*55)
    for nome,m in resultados.items():
        print(f"  {nome:<22} {m['auc_macro']:>6.4f} {m['f1_macro']:>6.4f} "
              f"{m['ks']:>6.4f} {m['accuracy']:>6.4f} {m['tempo_treino_s']:>6.1f}s")

    melhor = max(resultados.items(), key=lambda x: x[1]["auc_macro"])
    print(f"\n  Melhor AUC: {melhor[0]} ({melhor[1]['auc_macro']:.4f})")
    print("""
  RECOMENDACAO PARA PRODUCAO:
  -> Logistic Regression: inferencia <1ms, interpretavel, AUC competitivo
  -> XGBoost: melhor KS, ideal para scoring de risco juridico
  -> Random Forest: mais robusto a dados fora da distribuicao
    """)

    meta_json = {}
    for nome,m in resultados.items():
        meta_json[nome] = {k:float(v) if not isinstance(v,np.ndarray) else v.tolist()
                           for k,v in m.items()}
    with open(MODELS_DIR/"metricas_comparacao.json","w") as f:
        json.dump({"classes":classes,"modelos":meta_json},f,indent=2)

    rows = [{"Modelo":n,"AUC":f"{m['auc_macro']:.4f}","F1":f"{m['f1_macro']:.4f}",
             "KS":f"{m['ks']:.4f}","Accuracy":f"{m['accuracy']:.4f}",
             "Tempo(s)":f"{m['tempo_treino_s']:.1f}"} for n,m in resultados.items()]
    pd.DataFrame(rows).to_csv(DOCS_DIR/"tabela_metricas.csv",index=False)
    print(f"  Resultados salvos em: {MODELS_DIR}")
    print("\nPROXIMA ETAPA: python src/explainability/shap_analysis.py\n")
    return resultados, classes

if __name__ == "__main__":
    import sys
    tipo = next((a for a in sys.argv[1:] if a in ["tfidf","sbert","bert"]),"sbert")
    treinar(tipo)
