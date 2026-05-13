import pandas as pd, numpy as np, plotly.graph_objects as go, re
from plotly.subplots import make_subplots
from collections import Counter
from pathlib import Path
from loguru import logger
import warnings; warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DOCS_EDA = BASE_DIR / "docs" / "eda"
DOCS_EDA.mkdir(parents=True, exist_ok=True)

CORES = {"civil":"#2563EB","penal":"#DC2626","trabalhista":"#16A34A",
         "tributario":"#D97706","administrativo":"#7C3AED","consumidor":"#DB2777"}

STOPWORDS = {"de","da","do","das","dos","a","o","as","os","em","no","na","nos","nas",
             "por","para","com","sem","que","se","ao","e","ou","mas","sao","foi",
             "foram","ser","ha","nao","um","uma","seu","sua","seus","suas","este",
             "esta","esse","essa","recurso","especial","acordao","decisao","tribunal",
             "relator","ministro","camara","turma","provido","desprovido","negado",
             "concedido","mantido","autos","presentes","parte","partes"}

def carregar():
    p = DATA_RAW/"dataset_juridico.parquet"
    if not p.exists(): p = DATA_RAW/"dataset_juridico.csv"
    df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    df["n_chars"]    = df["texto"].str.len()
    df["n_palavras"] = df["texto"].str.split().str.len()
    df["riqueza"]    = df["texto"].apply(
        lambda t: len(set(t.lower().split()))/len(t.split()) if t.split() else 0)
    return df

def top_palavras(textos, n=20):
    todas = []
    for t in textos:
        ws = re.findall(r'\b[a-zA-Z]{4,}\b', t.lower())
        todas.extend([w for w in ws if w not in STOPWORDS])
    return Counter(todas).most_common(n)

def grafico_distribuicao(df):
    c = df["classe"].value_counts().reset_index()
    c.columns = ["classe","quantidade"]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1,cols=2,subplot_titles=("Contagem","Percentual"),
                        specs=[[{"type":"bar"},{"type":"pie"}]])
    fig.add_trace(go.Bar(x=c["classe"],y=c["quantidade"],
        marker_color=[CORES.get(x,"#64748b") for x in c["classe"]],
        text=c["quantidade"],textposition="outside"),row=1,col=1)
    fig.add_trace(go.Pie(labels=c["classe"],values=c["quantidade"],hole=0.4,
        marker_colors=[CORES.get(x,"#64748b") for x in c["classe"]]),row=1,col=2)
    fig.update_layout(title="Distribuicao das Classes Juridicas",
        template="plotly_white",height=450,showlegend=False)
    return fig

def grafico_comprimento(df):
    fig = make_subplots(rows=1,cols=2,
        subplot_titles=("Palavras por Classe","Riqueza Lexical"))
    for cls in sorted(df["classe"].unique()):
        sub = df[df["classe"]==cls]
        fig.add_trace(go.Box(y=sub["n_palavras"],name=cls,
            marker_color=CORES.get(cls,"#64748b"),showlegend=False),row=1,col=1)
        fig.add_trace(go.Violin(y=sub["riqueza"],name=cls,
            fillcolor=CORES.get(cls,"#64748b"),opacity=0.7,
            showlegend=False,box_visible=True),row=1,col=2)
    fig.update_layout(title="Comprimento e Riqueza Lexical",
        template="plotly_white",height=500)
    return fig

def grafico_palavras(df):
    classes = sorted(df["classe"].unique())
    fig = make_subplots(rows=2,cols=3,
        subplot_titles=[f"Top Palavras: {c.upper()}" for c in classes])
    for idx,cls in enumerate(classes):
        row,col = idx//3+1, idx%3+1
        pf = top_palavras(df[df["classe"]==cls]["texto"],15)
        if not pf: continue
        ps,fs = zip(*pf)
        fig.add_trace(go.Bar(x=list(fs[::-1]),y=list(ps[::-1]),orientation="h",
            marker_color=CORES.get(cls,"#64748b"),showlegend=False),row=row,col=col)
    fig.update_layout(title="Top Palavras por Classe",
        template="plotly_white",height=700)
    return fig

def grafico_heatmap(df):
    classes = sorted(df["classe"].unique())
    vocab = {c:set(p[0] for p in top_palavras(
        df[df["classe"]==c]["texto"],100)) for c in classes}
    n = len(classes)
    mat = np.zeros((n,n))
    for i,c1 in enumerate(classes):
        for j,c2 in enumerate(classes):
            if i==j: mat[i][j]=1.0
            else:
                inter=vocab[c1]&vocab[c2]; uni=vocab[c1]|vocab[c2]
                mat[i][j]=len(inter)/len(uni) if uni else 0
    fig = go.Figure(go.Heatmap(z=mat,x=classes,y=classes,
        colorscale="RdYlGn_r",text=np.round(mat,2),
        texttemplate="%{text}",zmin=0,zmax=1))
    fig.update_layout(title="Similaridade de Vocabulario (Jaccard)",
        template="plotly_white",height=500)
    return fig

def executar():
    print("\n"+"="*60+"\n  ETAPA 2 - EDA COM PLOTLY INTERATIVO\n"+"="*60)
    df = carregar()
    graficos = [
        ("01_distribuicao_classes.html", grafico_distribuicao),
        ("02_comprimento_textos.html",   grafico_comprimento),
        ("03_palavras_frequentes.html",  grafico_palavras),
        ("04_heatmap_vocabulario.html",  grafico_heatmap),
    ]
    for nome,func in graficos:
        print(f"  Gerando {nome}...", end=" ")
        try:
            func(df).write_html(str(DOCS_EDA/nome),include_plotlyjs="cdn")
            print("OK")
        except Exception as e:
            print(f"FALHOU: {e}")
    print(f"\nGraficos salvos em: {DOCS_EDA}")
    print("\nPROXIMA ETAPA: python src/preprocessing/limpeza_texto.py\n")

if __name__ == "__main__":
    executar()
