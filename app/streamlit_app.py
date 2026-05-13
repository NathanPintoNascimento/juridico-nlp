import sys, json, pickle, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "explainability"))

import streamlit as st

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models_saved"

st.set_page_config(page_title="Classificador Juridico NLP",page_icon="⚖️",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp{background:#050a14}
h1,h2,h3{font-family:'Inter',sans-serif!important}
.header-box{background:linear-gradient(135deg,#0d1b2e,#091525);
  border:1px solid #1e3a5f;border-radius:14px;padding:32px 36px;margin-bottom:24px}
.header-title{font-size:2.4rem;font-weight:700;color:#e8d5a3;margin:0 0 6px}
.header-sub{color:#64748b;font-size:0.9rem;margin:0}
.badge{background:#0f172a;border:1px solid #1e3a5f;border-radius:20px;
  padding:4px 12px;font-size:0.7rem;font-weight:500;color:#94a3b8;
  text-transform:uppercase;letter-spacing:.5px;margin-right:6px}
.result-card{border-radius:12px;padding:22px 26px;margin:14px 0;border:1px solid}
.gauge-bg{background:#1e293b;border-radius:6px;height:10px;overflow:hidden;margin:8px 0}
.gauge-fill{height:100%;border-radius:6px}
.prob-row{display:flex;align-items:center;gap:10px;margin:5px 0}
.prob-label{width:130px;font-size:0.82rem;color:#94a3b8;text-align:right;flex-shrink:0}
.prob-bg{flex:1;background:#0f172a;border-radius:4px;height:8px;overflow:hidden}
.prob-fill{height:100%;border-radius:4px}
.prob-pct{width:42px;text-align:right;font-size:0.8rem;color:#64748b;flex-shrink:0;font-family:monospace}
.word-table{width:100%;border-collapse:collapse;font-size:0.85rem}
.word-table th{background:#0d1520;color:#64748b;font-size:0.7rem;text-transform:uppercase;
  letter-spacing:.5px;padding:8px 12px;text-align:left;border-bottom:1px solid #1a2744}
.word-table td{padding:7px 12px;border-bottom:1px solid #0f1d30;color:#94a3b8}
.pos{color:#22c55e;font-weight:600;font-family:monospace}
.neg{color:#ef4444;font-weight:600;font-family:monospace}
.stTextArea textarea{background:#0d1b2e!important;border:1px solid #1e3a5f!important;
  color:#c8d8f0!important;border-radius:10px!important;font-family:Georgia,serif!important;
  font-size:0.92rem!important;line-height:1.7!important}
.stButton>button{background:linear-gradient(135deg,#1d4ed8,#1e40af)!important;
  color:#eff6ff!important;border:1px solid #2563eb!important;border-radius:10px!important;
  font-weight:600!important;width:100%!important}
[data-testid="stSidebar"]{background:#080f1a!important}
</style>
""", unsafe_allow_html=True)

CORES = {"civil":{"hex":"#2563EB","bg":"#0d1d3e","border":"#1e3a7a"},
         "penal":{"hex":"#DC2626","bg":"#2a0a0a","border":"#5a1a1a"},
         "trabalhista":{"hex":"#16A34A","bg":"#052011","border":"#0a4022"},
         "tributario":{"hex":"#D97706","bg":"#2a1500","border":"#5a3000"},
         "administrativo":{"hex":"#7C3AED","bg":"#1a0a30","border":"#3a1a60"},
         "consumidor":{"hex":"#DB2777","bg":"#2a0518","border":"#5a0a30"}}
ICONES = {"civil":"⚖️","penal":"🔒","trabalhista":"👷",
          "tributario":"💰","administrativo":"🏛️","consumidor":"🛡️"}
DESCS  = {"civil":"Contratos, familia, divorcio, responsabilidade civil",
          "penal":"Crimes, penas, habeas corpus, tribunal do juri",
          "trabalhista":"CLT, FGTS, rescisao, horas extras, vinculo",
          "tributario":"Impostos, execucao fiscal, ICMS, parcelamento",
          "administrativo":"Licitacao, servidores, improbidade, concurso",
          "consumidor":"CDC, dano moral, produto defeituoso, plano de saude"}
EXEMPLOS = {
    "🔒 Penal": ("HABEAS CORPUS. Prisao preventiva decretada por suposta pratica de "
                 "roubo qualificado pelo emprego de arma de fogo. A defesa alega ausencia "
                 "de fundamentacao concreta acerca do risco a ordem publica. O paciente "
                 "responde primariamente e possui residencia fixa. Pedido de liberdade "
                 "provisoria com medidas cautelares diversas da prisao, nos termos do "
                 "art. 319 do Codigo de Processo Penal."),
    "👷 Trabalhista": ("RECURSO DE REVISTA. Rescisao indireta do contrato de trabalho por "
                       "descumprimento de obrigacoes patronais. O empregador deixou de recolher "
                       "o FGTS por periodo superior a doze meses consecutivos e atrasou o "
                       "pagamento de salarios reiteradamente. A reclamante pleiteia verbas "
                       "rescisorias como demissao sem justa causa, incluindo aviso previo e "
                       "multa de 40% sobre o FGTS."),
    "💰 Tributario": ("EXECUCAO FISCAL. ICMS. O contribuinte impugna a certidao de divida ativa "
                      "alegando que o credito tributario esta prescrito, pois o Fisco deixou de "
                      "promover a cobranca no prazo quinquenal do art. 174 do CTN. "
                      "Subsidiariamente, sustenta excesso no lancamento por aplicacao "
                      "equivocada da aliquota de substituicao tributaria."),
    "🛡️ Consumidor": ("ACAO DE INDENIZACAO. Negativacao indevida do nome do consumidor nos "
                      "cadastros do SPC e SERASA. O autor comprova que o debito foi quitado "
                      "antes da inscricao. O fornecedor nao realizou a baixa tempestiva, "
                      "causando dano moral in re ipsa ao consumidor, independentemente de "
                      "comprovacao de prejuizo concreto."),
    "🏛️ Administrativo": ("MANDADO DE SEGURANCA. Concurso publico. Candidato aprovado dentro "
                           "do numero de vagas previstas no edital que nao foi nomeado. "
                           "A Administracao alega discricionariedade. Contudo, o STF e STJ "
                           "consolidaram que candidato aprovado dentro das vagas tem direito "
                           "subjetivo a nomeacao, configurando ato vinculado."),
}

@st.cache_resource(show_spinner=False)
def carregar():
    art = {}
    for nome in ["logistic_regression","random_forest","xgboost"]:
        p = MODELS_DIR/f"{nome}.pkl"
        if p.exists():
            with open(p,"rb") as f: art[nome] = pickle.load(f)
    for nome in ["label_encoder","tfidf_vectorizer","svd_lsa"]:
        p = MODELS_DIR/f"{nome}.pkl"
        if p.exists():
            with open(p,"rb") as f: art[nome] = pickle.load(f)
    mp = MODELS_DIR/"metricas_comparacao.json"
    if mp.exists():
        with open(mp) as f: art["metricas"] = json.load(f)
    return art

def classificar(texto, art, modelo_nome):
    le  = art.get("le") or art.get("label_encoder")
    vec = art.get("vectorizer") or art.get("tfidf_vectorizer")
    svd = art.get("svd") or art.get("svd_lsa")
    modelo = art.get(modelo_nome)
    if not all([le,vec,svd,modelo]):
        return {"erro":"Modelos nao carregados. Execute o pipeline primeiro."}
    classes = le.classes_.tolist()
    X = svd.transform(vec.transform([texto]))
    probs = modelo.predict_proba(X)[0]
    ci = int(np.argmax(probs))
    try:    classe = le.inverse_transform([ci])[0]
    except: classe = classes[ci]
    return {"classe":classe,"confianca":float(probs[ci]),
            "probabilidades":{c:float(p) for c,p in zip(classes,probs)}}

def explicar(texto, art):
    le  = art.get("le") or art.get("label_encoder")
    vec = art.get("vectorizer") or art.get("tfidf_vectorizer")
    svd = art.get("svd") or art.get("svd_lsa")
    modelo = art.get("logistic_regression")
    if not all([le,vec,svd,modelo]):
        return {"todas_palavras":[]}
    classes = le.classes_.tolist()
    X_svd = svd.transform(vec.transform([texto]))
    probs = modelo.predict_proba(X_svd)[0]
    ci = int(np.argmax(probs))
    X_tf = vec.transform([texto]).toarray()[0]
    fn = vec.get_feature_names_out()
    nz = np.nonzero(X_tf)[0]
    imps = [(fn[i],float(X_tf[i])) for i in nz if len(fn[i])>2]
    imps.sort(key=lambda x:x[1],reverse=True)
    return {"classe_predita":classes[ci],"confianca":float(probs[ci]),
            "prob_por_classe":{c:float(p) for c,p in zip(classes,probs)},
            "palavras_positivas":imps[:10],"palavras_negativas":[],
            "todas_palavras":imps}

def highlight(texto, palavras_scores):
    if not palavras_scores:
        return f'<div style="padding:16px;background:#0a1628;border-radius:8px;color:#94a3b8;font-family:Georgia,serif;line-height:1.9">{texto}</div>'
    scores  = {p.lower():float(s) for p,s in palavras_scores}
    max_abs = max(abs(s) for s in scores.values()) if scores else 1.0
    parts = []
    for w in texto.split():
        limpa = re.sub(r'[^\w]','',w.lower())
        s = scores.get(limpa,0)
        if abs(s)<max_abs*0.05:
            parts.append(f'<span style="color:#475569">{w}</span>')
        elif s>0:
            a = min(0.9,abs(s)/max_abs)
            parts.append(f'<span style="background:rgba(34,197,94,{0.15+a*0.35:.2f});color:#f0fdf4;border-radius:3px;padding:1px 4px;font-weight:600" title="+{s:.4f}">{w}</span>')
        else:
            a = min(0.9,abs(s)/max_abs)
            parts.append(f'<span style="background:rgba(239,68,68,{0.1+a*0.25:.2f});color:#fff1f2;border-radius:3px;padding:1px 4px" title="{s:.4f}">{w}</span>')
    return ('<div style="font-family:Georgia,serif;font-size:0.95rem;line-height:1.9;'
            'padding:16px;background:#0a1628;border-radius:8px;border:1px solid #1e3a5f">'
            +' '.join(parts)+'</div>')

def main():
    with st.spinner("Carregando modelos..."):
        art = carregar()
    modelos_ok = any(m in art for m in ["logistic_regression","random_forest","xgboost"])
    modelos_disp = [m for m in ["logistic_regression","random_forest","xgboost"] if m in art]

    st.markdown("""
    <div class="header-box">
      <div class="header-title">⚖️ Classificador Juridico NLP</div>
      <div class="header-sub">Classifica textos juridicos em 6 areas do direito brasileiro</div>
      <div style="margin-top:14px">
        <span class="badge">🇧🇷 Portugues Juridico</span>
        <span class="badge">🤖 BERTimbau</span>
        <span class="badge">🔍 LIME Explainer</span>
        <span class="badge">6 Classes</span>
      </div>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Configuracoes")
        modelo_sel = st.selectbox("Modelo:",options=modelos_disp,
            format_func=lambda x:{"logistic_regression":"Logistic Regression",
                                   "random_forest":"Random Forest",
                                   "xgboost":"XGBoost"}.get(x,x)) if modelos_disp else None
        mostrar_exp = st.toggle("🔍 Mostrar explicacao",value=True)
        st.markdown("---")
        st.markdown("### 💡 Exemplos")
        texto_ex = None
        for label,texto in EXEMPLOS.items():
            if st.button(label,use_container_width=True,key=f"ex_{label}"):
                texto_ex = texto
        st.markdown("---")
        if "metricas" in art and modelo_sel:
            m = art["metricas"].get("modelos",{}).get(modelo_sel,{})
            if m:
                st.markdown("### 📊 Metricas")
                c1,c2 = st.columns(2)
                c1.metric("AUC",  f"{float(m.get('auc_macro',0)):.3f}")
                c2.metric("F1",   f"{float(m.get('f1_macro',0)):.3f}")
                st.metric("KS",   f"{float(m.get('ks',0)):.3f}")

    col1,col2 = st.columns([1.1,0.9],gap="large")
    with col1:
        st.markdown("#### 📝 Cole o texto juridico")
        val = texto_ex if texto_ex else ""
        texto = st.text_area("texto",value=val,height=300,label_visibility="collapsed",
            placeholder="Cole aqui uma ementa, decisao, recurso ou qualquer texto juridico...")
        nw = len(texto.split()) if texto.strip() else 0
        st.markdown(f'<div style="text-align:right;color:#334155;font-size:0.75rem">{nw} palavras</div>',
                    unsafe_allow_html=True)
        classificar_btn = st.button("🔍 Classificar Texto",type="primary",
                                     disabled=not(texto.strip() and modelos_ok))
        if not modelos_ok:
            st.info("⚠️ **Modelos nao encontrados.**\n\nExecute:\n```\npython run_all.py --no-bert\n```")

    with col2:
        if classificar_btn and texto.strip() and modelos_ok:
            with st.spinner("Classificando..."):
                res = classificar(texto, art, modelo_sel)
            if "erro" in res:
                st.error(res["erro"])
            else:
                classe = res["classe"]; conf = res["confianca"]; probs = res["probabilidades"]
                cores = CORES.get(classe,{"hex":"#64748b","bg":"#0f172a","border":"#1e293b"})
                icone = ICONES.get(classe,"⚖️"); desc = DESCS.get(classe,"")
                st.markdown(f"""
                <div class="result-card" style="background:{cores['bg']};border-color:{cores['border']}">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
                    <span style="font-size:2.2rem">{icone}</span>
                    <div>
                      <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.8px;color:#64748b">Area do Direito</div>
                      <div style="font-size:1.8rem;font-weight:700;color:{cores['hex']};line-height:1.1">{classe.upper()}</div>
                      <div style="font-size:.8rem;color:#64748b">{desc}</div>
                    </div>
                  </div>
                  <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:14px">
                    <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;color:#64748b">Confianca</div>
                    <div class="gauge-bg"><div class="gauge-fill" style="width:{conf*100:.1f}%;background:{cores['hex']}"></div></div>
                    <div style="font-size:1.4rem;font-weight:700;color:{cores['hex']}">{conf:.1%}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.markdown("#### 📊 Probabilidades")
                prob_html = ""
                for cls,prob in sorted(probs.items(),key=lambda x:x[1],reverse=True):
                    ch = CORES.get(cls,{"hex":"#64748b"})["hex"]
                    fw = "font-weight:600;color:#e2e8f0" if cls==classe else ""
                    prob_html += f"""<div class="prob-row">
                      <div class="prob-label" style="{fw}">{ICONES.get(cls,"")} {cls}</div>
                      <div class="prob-bg"><div class="prob-fill" style="width:{prob*100:.1f}%;background:{ch};opacity:{0.4+prob*0.6:.2f}"></div></div>
                      <div class="prob-pct">{prob:.1%}</div></div>"""
                st.markdown(prob_html, unsafe_allow_html=True)

        elif not texto.strip():
            st.markdown("""<div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:380px;color:#1e293b">
                <div style="font-size:4rem;opacity:.3">⚖️</div>
                <div style="font-size:.9rem;color:#334155;text-align:center;margin-top:12px">
                Cole um texto juridico ao lado<br>e clique em Classificar</div></div>""",
                unsafe_allow_html=True)

    if classificar_btn and texto.strip() and modelos_ok and mostrar_exp:
        st.markdown("---")
        st.markdown("### 🔍 Explicabilidade — Por que essa classificacao?")
        st.markdown("""<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
            padding:10px 14px;margin-bottom:14px;font-size:.82rem;color:#64748b">
            <b style="color:#94a3b8">Como ler:</b>
            <span style="background:rgba(34,197,94,.3);color:#f0fdf4;border-radius:3px;padding:1px 6px;margin:0 4px">verde</span>
            = aumenta probabilidade &nbsp;|&nbsp;
            <span style="background:rgba(239,68,68,.25);color:#fff1f2;border-radius:3px;padding:1px 6px;margin:0 4px">vermelho</span>
            = diminui &nbsp;|&nbsp; <span style="color:#475569">cinza</span> = neutro
            </div>""", unsafe_allow_html=True)
        with st.spinner("Calculando explicacao..."):
            exp = explicar(texto, art)
        col_hl,col_tb = st.columns([1.2,0.8],gap="large")
        with col_hl:
            st.markdown("**Texto destacado:**")
            st.markdown(highlight(texto,exp.get("todas_palavras",[])),unsafe_allow_html=True)
        with col_tb:
            st.markdown("**Top palavras influentes:**")
            todas = exp.get("todas_palavras",[])[:15]
            if todas:
                max_s = max(abs(s) for _,s in todas) if todas else 1.0
                rows = ""
                for palavra,score in todas:
                    cls = "pos" if score>0 else "neg"
                    sinal = "+" if score>0 else ""
                    cor = "#22c55e" if score>0 else "#ef4444"
                    bw = int(abs(score)/max_s*80)
                    rows += f"""<tr>
                      <td><code style="color:#94a3b8;font-size:.82rem">{palavra}</code></td>
                      <td><div style="background:#1e293b;border-radius:3px;height:6px;width:90px">
                        <div style="background:{cor};width:{bw}%;height:100%;border-radius:3px;opacity:.7"></div></div></td>
                      <td class="{cls}">{sinal}{score:.4f}</td></tr>"""
                st.markdown(f"""<table class="word-table">
                  <thead><tr><th>Palavra</th><th>Impacto</th><th>Score</th></tr></thead>
                  <tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="text-align:center;color:#1e293b;font-size:.75rem">Projeto educacional · NLP Juridico com BERTimbau e SHAP/LIME</div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
