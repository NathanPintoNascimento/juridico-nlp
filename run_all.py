import sys, os, argparse, subprocess
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

ETAPAS = {
    "etapa1": {"nome": "Coleta de Dados",    "script": "src/coleta_dados.py"},
    "etapa2": {"nome": "EDA com Plotly",     "script": "src/eda.py"},
    "etapa3": {"nome": "Pre-processamento",  "script": "src/preprocessing/limpeza_texto.py"},
    "etapa4": {"nome": "Embeddings",         "script": "src/models/embeddings.py"},
    "etapa5": {"nome": "Treinamento",        "script": "src/models/treinamento.py"},
    "etapa6": {"nome": "Explicabilidade",    "script": "src/explainability/shap_analysis.py"},
}

def rodar_etapa(key, extra=None):
    e = ETAPAS[key]
    script = BASE_DIR / e["script"]
    if not script.exists():
        print(f"Script nao encontrado: {script}"); return False
    cmd = [sys.executable, str(script)] + (extra or [])
    print(f"\n{'='*60}\n  {e['nome']}\n{'='*60}")
    return subprocess.run(cmd, cwd=str(BASE_DIR)).returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bert",    action="store_true")
    parser.add_argument("--from",       dest="from_etapa", default=None, choices=list(ETAPAS.keys()))
    parser.add_argument("--only",       dest="only_etapa", default=None, choices=list(ETAPAS.keys()))
    args = parser.parse_args()
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models_saved", exist_ok=True)
    todas = list(ETAPAS.keys())
    if args.only_etapa:   etapas = [args.only_etapa]
    elif args.from_etapa: etapas = todas[todas.index(args.from_etapa):]
    else:                 etapas = todas
    extra_map = {"etapa4": ["--no-bert"] if args.no_bert else []}
    resultados = {}
    for k in etapas:
        ok = rodar_etapa(k, extra_map.get(k, []))
        resultados[k] = ok
        if not ok:
            print(f"\nFalha em {k}. Verifique o erro acima."); break
    print("\n" + "="*60 + "\nRESUMO:")
    for k, ok in resultados.items():
        print(f"  {'OK' if ok else 'FALHOU'}  {ETAPAS[k]['nome']}")
    if all(resultados.values()):
        print("\nPIPELINE CONCLUIDO!\nPROXIMO: streamlit run app/streamlit_app.py\n")

if __name__ == "__main__":
    main()
