import sys, pytest
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR/"src"))
sys.path.insert(0, str(BASE_DIR/"src"/"preprocessing"))

TEXTO_PENAL = "HABEAS CORPUS. Prisao preventiva. Ausencia de fundamentacao. O paciente responde por roubo qualificado pelo emprego de arma de fogo. Pedido de liberdade provisoria."
TEXTO_TRAB  = "RECURSO DE REVISTA. Rescisao indireta. FGTS nao recolhido. Horas extras nao pagas. CLT art 483."
TEXTO_TRIB  = "EXECUCAO FISCAL. ICMS. Substituicao tributaria. Prescricao quinquenal. Art 174 CTN."

class TestLimpeza:
    def test_importa(self):
        from limpeza_texto import LimpadorJuridico
        assert LimpadorJuridico is not None

    def test_limpa_classico(self):
        from limpeza_texto import LimpadorJuridico
        r = LimpadorJuridico(modo="classico").limpar(TEXTO_PENAL)
        assert isinstance(r,str) and len(r)>10

    def test_neural_maior(self):
        from limpeza_texto import LimpadorJuridico
        lc = LimpadorJuridico(modo="classico").limpar(TEXTO_PENAL)
        ln = LimpadorJuridico(modo="neural").limpar(TEXTO_PENAL)
        assert len(ln) >= len(lc)

    def test_vazio(self):
        from limpeza_texto import LimpadorJuridico
        l = LimpadorJuridico()
        assert l.limpar("") == "" and l.limpar("   ") == ""

    def test_batch(self):
        from limpeza_texto import LimpadorJuridico
        r = LimpadorJuridico().limpar_batch([TEXTO_PENAL,TEXTO_TRAB,TEXTO_TRIB])
        assert len(r)==3 and all(isinstance(x,str) for x in r)

class TestColeta:
    def test_importa(self):
        import coleta_dados; assert coleta_dados is not None

    def test_gera_sinteticos(self):
        from coleta_dados import gerar_dados_sinteticos
        df = gerar_dados_sinteticos(n_por_classe=5)
        assert len(df) == 30

    def test_classes_corretas(self):
        from coleta_dados import gerar_dados_sinteticos
        df = gerar_dados_sinteticos(n_por_classe=3)
        assert set(df["classe"].unique()) == {"civil","penal","trabalhista","tributario","administrativo","consumidor"}

    def test_textos_nao_vazios(self):
        from coleta_dados import gerar_dados_sinteticos
        df = gerar_dados_sinteticos(n_por_classe=3)
        assert (df["texto"].str.len()>50).all()

@pytest.mark.integration
class TestIntegracao:
    def test_modelo_carrega(self):
        import pickle
        p = BASE_DIR/"models_saved"/"logistic_regression.pkl"
        if not p.exists(): pytest.skip("Pipeline nao executado")
        with open(p,"rb") as f: m = pickle.load(f)
        assert hasattr(m,"predict_proba")

    def test_predicao_6_classes(self):
        import pickle, numpy as np
        for nome in ["logistic_regression.pkl","tfidf_vectorizer.pkl","svd_lsa.pkl"]:
            if not (BASE_DIR/"models_saved"/nome).exists():
                pytest.skip("Pipeline nao executado")
        with open(BASE_DIR/"models_saved"/"logistic_regression.pkl","rb") as f: m=pickle.load(f)
        with open(BASE_DIR/"models_saved"/"tfidf_vectorizer.pkl","rb") as f:   v=pickle.load(f)
        with open(BASE_DIR/"models_saved"/"svd_lsa.pkl","rb") as f:            s=pickle.load(f)
        X = s.transform(v.transform([TEXTO_PENAL]))
        probs = m.predict_proba(X)
        assert probs.shape==(1,6) and abs(probs.sum()-1.0)<1e-5
