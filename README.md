#  Classificador de Textos Jurídicos com NLP

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BERTimbau-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-189AB4?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilidade-4CAF50?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> Projeto de classificação automática de documentos jurídicos brasileiros usando embeddings com BERTimbau, modelos de ML clássicos e interface interativa via Streamlit.

---

##  Sobre o Projeto

Este projeto classifica textos jurídicos (acórdãos, decisões, petições) extraídos de fontes abertas do STJ e STF, utilizando técnicas modernas de NLP com transformers em português.

O usuário cola qualquer texto jurídico na interface e recebe:
- ✅ A **categoria predita** (ex: Direito Penal, Trabalhista, Civil...)
- 🔍 As **palavras mais influentes** na decisão (via SHAP)
- 📊 O **score de confiança** da predição

---

##  Fonte dos Dados

| Fonte | Tipo | Link |
|---|---|---|
| STJ | Acórdãos e decisões monocráticas | [stj.jus.br](https://www.stj.jus.br/sites/portalp/Processos/Consulta-Processual) |
| STF | Decisões e jurisprudências | [stf.jus.br](https://jurisprudencia.stf.jus.br/) |
| JusBrasil | Agregador de tribunais estaduais | [jusbrasil.com.br](https://www.jusbrasil.com.br/) |

Os dados foram coletados via scraping/API pública e estruturados em formato `.csv` com colunas `texto` e `classe`.

---

##  Estrutura do Projeto

```
juridico-nlp/
│
├── app/
│   └── streamlit_app.py       # Interface Streamlit
│
├── data/
│   ├── raw/                   # Dados brutos coletados
│   └── processed/             # Dados limpos e tokenizados
│
├── notebooks/
│   ├── 01_eda.ipynb           # Análise exploratória com Plotly
│   ├── 02_preprocessing.ipynb # Limpeza e pré-processamento
│   ├── 03_embeddings.ipynb    # Geração de embeddings BERTimbau
│   └── 04_models.ipynb        # Treino e comparação de modelos
│
├── models/                    # Modelos treinados (.pkl, .pt)
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Comparação de Modelos

| Modelo | AUC | F1 (macro) | KS | Tempo de Treino |
|---|---|---|---|---|
| Logistic Regression | 0.91 | 0.87 | 0.74 | ~12s |
| Random Forest | 0.93 | 0.89 | 0.77 | ~4min |
| **XGBoost** | **0.95** | **0.92** | **0.81** | ~6min |

###  Modelo em Produção: XGBoost

O XGBoost foi escolhido por apresentar o melhor equilíbrio entre AUC, F1 e KS. Diferente do Random Forest, ele lida melhor com dados desbalanceados (comum em datasets jurídicos) e oferece suporte nativo a SHAP para explicabilidade.

---

##  Como Rodar o Projeto

### 1. Clone o repositório
```bash
git clone https://github.com/NathanPintoNascimento/juridico-nlp.git
cd juridico-nlp
```

### 2. Crie e ative o ambiente virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Rode a interface Streamlit
```bash
python -m streamlit run app/streamlit_app.py
```

---

##  Stack Técnica

- **Embeddings**: `sentence-transformers` + `neuralmind/bert-base-portuguese-cased` (BERTimbau)
- **Modelos**: Logistic Regression, Random Forest, XGBoost (via scikit-learn e xgboost)
- **Explicabilidade**: SHAP (SHapley Additive exPlanations)
- **EDA**: Plotly Express + Plotly Graph Objects
- **Interface**: Streamlit
- **Pré-processamento**: spaCy (`pt_core_news_lg`), NLTK, regex

---

##  O Que Não Funcionou e Por Quê

| Tentativa | Problema | Solução Adotada |
|---|---|---|
| Fine-tuning completo do BERTimbau | Custo computacional inviável sem GPU | Usado apenas como extrator de features (embeddings fixos) |
| TF-IDF puro como representação | Perda semântica em textos longos jurídicos | Substituído por sentence-transformers |
| Coleta via scraping direto do STJ | Bloqueio por rate limit e CAPTCHA | Uso da API pública e datasets já disponíveis no Kaggle |
| Balanceamento com SMOTE | Geração de amostras sintéticas incoerentes para texto | Substituído por `class_weight='balanced'` nos modelos |

---

##  Próximos Passos

- [ ] Fine-tuning do BERTimbau em GPU (Google Colab Pro / Kaggle)
- [ ] Adicionar mais classes jurídicas (atualmente 5 categorias)
- [ ] Deploy na Streamlit Cloud ou Hugging Face Spaces
- [ ] Pipeline de atualização automática dos dados via API do STJ
- [ ] Adicionar suporte a upload de PDF na interface

---

##  Autor

**Nathan Pinto Nascimento**  
Estudante de Sistemas de Informação  
[![GitHub](https://img.shields.io/badge/GitHub-NathanPintoNascimento-181717?style=flat&logo=github)](https://github.com/NathanPintoNascimento)

---

##  Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.# juridico-nlp
