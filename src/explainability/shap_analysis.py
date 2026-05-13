import numpy as np, pandas as pd, pickle, json, re, warnings
from pathlib import Path
from loguru import logger
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR/"data"/"processed"
MODELS_DIR     = BASE_DIR/"models_saved"
DOCS_SHAP      = BASE_DIR/"docs"/"shap"
DOCS_SHAP.mkdir(parents=True, exist_ok=True)

CORES = {"civil":"#2563EB","penal":"#DC2626","trabalhista":"#16A34A",
         "tributario":"#D97706","administrativo":"#7C3AED","consumidor":"#DB2777"}

def calcular_shap_global(n_samples=200):
    try: import shap
    except ImportError:
        print("  SHAP nao instalado. Pulando analise global.")
        return None
    lr_p  = MODELS_DIR/"logistic_regression.pkl"
    vec_p = MODELS_DIR/"tfidf_vectorizer.pkl"
    svd_p = MODELS_DIR/"svd_lsa.pkl"
    if not all(p.exists() for p in [lr_p,vec_p]):
        print("  Modelos nao encontrados. Execute o treinamento primeiro.")
        return None
    with open(lr_p,"rb") as f:  modelo = pickle.load(f)
    with open(vec_p,"rb") as f: vectorizer = pickle.load(f)
    with open(MODELS_DIR/"label_encoder.pkl","rb") as f: le = pickle.load(f)
    with open(DATA_PROCESSED/"splits.pkl","rb") as f: splits = pickle.load(f)
    classes = le.classes_.tolist()
    X_te = vectorizer.transform(splits["X_test"][:n_samples])
    X_tr = vectorizer.transform(splits["X_train"])
    bg = shap.utils.sample(X_tr, min(100,X_tr.shape[0]))
    explainer = shap.LinearExplainer(modelo, bg)
    shap_values = explainer.shap_values(X_te)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  SHAP calculado: {len(shap_values)} classes x {X_te.shape[0]} amostras")
    return {"shap_values":shap_values,"feature_names":feature_names,
            "classes":classes,"X_tfidf":X_te}

def plotar_bar_global(shap_data):
    if shap_data is None: return
    classes       = shap_data["classes"]
    feature_names = shap_data["feature_names"]
    shap_values   = shap_data["shap_values"]
    fig, axes = plt.subplots(2,3,figsize=(18,10))
    axes = axes.flatten()
    for idx,(classe,ax) in enumerate(zip(classes,axes)):
        sv = shap_values[idx]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:15]
        top_w = feature_names[top_idx][::-1]
        top_s = mean_abs[top_idx][::-1]
        color = CORES.get(classe,"#64748b")
        ax.barh(range(len(top_w)),top_s,color=color,alpha=0.8)
        ax.set_yticks(range(len(top_w))); ax.set_yticklabels(top_w,fontsize=8)
        ax.set_title(classe.upper(),fontsize=11,fontweight="bold",color=color)
        ax.set_xlabel("|SHAP| medio",fontsize=8)
        ax.grid(axis="x",alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    plt.suptitle("SHAP Global - Palavras Mais Influentes por Classe",
                 fontsize=14,fontweight="bold",y=1.01)
    plt.tight_layout()
    path = DOCS_SHAP/"shap_global_bar.png"
    plt.savefig(str(path),dpi=150,bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {path}")

def explicar_fallback(texto):
    vec_p = MODELS_DIR/"tfidf_vectorizer.pkl"
    lr_p  = MODELS_DIR/"logistic_regression.pkl"
    svd_p = MODELS_DIR/"svd_lsa.pkl"
    if not all(p.exists() for p in [vec_p,lr_p]):
        return {"classe_predita":"N/A","confianca":0.0,"prob_por_classe":{},
                "palavras_positivas":[],"palavras_negativas":[],"todas_palavras":[],"classes":[]}
    with open(vec_p,"rb") as f: vec = pickle.load(f)
    with open(lr_p,"rb") as f:  modelo = pickle.load(f)
    with open(svd_p,"rb") as f: svd = pickle.load(f)
    with open(MODELS_DIR/"label_encoder.pkl","rb") as f: le = pickle.load(f)
    classes = le.classes_.tolist()
    X = svd.transform(vec.transform([texto]))
    probs = modelo.predict_proba(X)[0]
    ci = int(np.argmax(probs))
    classe = classes[ci]
    X_tf = vec.transform([texto]).toarray()[0]
    fn = vec.get_feature_names_out()
    nz = np.nonzero(X_tf)[0]
    imps = [(fn[i],float(X_tf[i])) for i in nz if len(fn[i])>2]
    imps.sort(key=lambda x:x[1],reverse=True)
    return {"classe_predita":classe,"confianca":float(probs[ci]),
            "prob_por_classe":{c:float(p) for c,p in zip(classes,probs)},
            "palavras_positivas":imps[:10],"palavras_negativas":[],
            "todas_palavras":imps,"classes":classes}

def explicar_texto_lime(texto):
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return explicar_fallback(texto)
    vec_p = MODELS_DIR/"tfidf_vectorizer.pkl"
    lr_p  = MODELS_DIR/"logistic_regression.pkl"
    svd_p = MODELS_DIR/"svd_lsa.pkl"
    if not all(p.exists() for p in [vec_p,lr_p,svd_p]):
        return explicar_fallback(texto)
    with open(vec_p,"rb") as f: vec = pickle.load(f)
    with open(lr_p,"rb") as f:  modelo = pickle.load(f)
    with open(svd_p,"rb") as f: svd = pickle.load(f)
    with open(MODELS_DIR/"label_encoder.pkl","rb") as f: le = pickle.load(f)
    classes = le.classes_.tolist()
    def predict_fn(textos):
        return modelo.predict_proba(svd.transform(vec.transform(textos)))
    probs = predict_fn([texto])[0]
    ci = int(np.argmax(probs))
    explainer = LimeTextExplainer(class_names=classes,bow=True,random_state=42)
    exp = explainer.explain_instance(texto,predict_fn,num_features=20,
                                     num_samples=500,top_labels=3)
    palavras = exp.as_list(label=ci)
    palavras.sort(key=lambda x:abs(x[1]),reverse=True)
    pos = [(p,v) for p,v in palavras if v>0]
    neg = [(p,v) for p,v in palavras if v<0]
    return {"classe_predita":classes[ci],"confianca":float(probs[ci]),
            "prob_por_classe":{c:float(p) for c,p in zip(classes,probs)},
            "palavras_positivas":pos[:10],"palavras_negativas":neg[:5],
            "todas_palavras":palavras,"classes":classes}

def gerar_html_highlight(texto, palavras_scores):
    if not palavras_scores:
        return f'<div style="padding:16px;background:#0f172a;border-radius:8px;color:#94a3b8">{texto}</div>'
    scores  = {p.lower():float(s) for p,s in palavras_scores}
    max_abs = max(abs(s) for s in scores.values()) if scores else 1.0
    parts   = []
    for palavra in texto.split():
        limpa = re.sub(r'[^\w]','',palavra.lower())
        score = scores.get(limpa,0)
        if abs(score) < max_abs*0.05:
            parts.append(f'<span style="color:#475569">{palavra}</span>')
        elif score > 0:
            a = min(0.9,abs(score)/max_abs)
            parts.append(f'<span style="background:rgba(34,197,94,{0.15+a*0.35:.2f});'
                         f'color:#f0fdf4;border-radius:3px;padding:1px 4px;font-weight:600"'
                         f' title="+{score:.4f}">{palavra}</span>')
        else:
            a = min(0.9,abs(score)/max_abs)
            parts.append(f'<span style="background:rgba(239,68,68,{0.1+a*0.25:.2f});'
                         f'color:#fff1f2;border-radius:3px;padding:1px 4px"'
                         f' title="{score:.4f}">{palavra}</span>')
    return ('<div style="font-family:Georgia,serif;font-size:0.95rem;line-height:1.9;'
            'padding:16px;background:#0a1628;border-radius:8px;border:1px solid #1e3a5f">'
            + " ".join(parts) + '</div>')

def executar():
    print("\n"+"="*60+"\n  ETAPA 6 - EXPLICABILIDADE SHAP + LIME\n"+"="*60)
    print("\n  Calculando SHAP global...")
    shap_data = calcular_shap_global(n_samples=200)
    plotar_bar_global(shap_data)

    exemplos = {
        "penal": ("HABEAS CORPUS. Prisao preventiva. Ausencia de fundamentacao concreta "
                  "acerca do risco a ordem publica. O paciente responde por roubo qualificado "
                  "pelo emprego de arma de fogo. Pedido de liberdade provisoria com medidas "
                  "cautelares diversas da prisao. Art. 319 do Codigo de Processo Penal."),
        "trabalhista": ("RECURSO DE REVISTA. Rescisao indireta do contrato de trabalho. "
                        "Nao recolhimento do FGTS por periodo superior a doze meses. "
                        "Pedido de verbas rescisorias como demissao sem justa causa."),
        "tributario": ("EXECUCAO FISCAL. ICMS. Substituicao tributaria. "
                       "Contribuinte pleiteia restituicao de tributo pago a maior "
                       "em decorrencia de aliquota aplicada incorretamente."),
    }
    print("\n  Gerando exemplos LIME...")
    for classe_esp, texto in exemplos.items():
        print(f"\n  -> {classe_esp}...", end=" ")
        res = explicar_texto_lime(texto)
        print(f"Predito: {res['classe_predita']} ({res['confianca']:.1%})")
        print(f"     Top palavras: {[p for p,_ in res['palavras_positivas'][:5]]}")
        html = gerar_html_highlight(texto, res["todas_palavras"])
        path = DOCS_SHAP/f"exemplo_{classe_esp}.html"
        with open(path,"w",encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>LIME - {classe_esp}</title>
<style>body{{background:#020617;padding:20px;color:#e2e8f0;font-family:sans-serif}}</style>
</head><body>
<h2 style="color:#c8a96e">Exemplo: {classe_esp.upper()}</h2>
<p><b>Predicao:</b> {res['classe_predita'].upper()} | <b>Confianca:</b> {res['confianca']:.1%}</p>
{html}
<p style="margin-top:20px;color:#64748b;font-size:0.8rem">
Verde = aumenta probabilidade | Vermelho = diminui probabilidade</p>
</body></html>""")
        print(f"     HTML: {path}")

    print(f"\n  Graficos SHAP em: {DOCS_SHAP}")
    print("\nPROXIMA ETAPA: streamlit run app/streamlit_app.py\n")

if __name__ == "__main__":
    import os; os.makedirs("logs",exist_ok=True)
    executar()
