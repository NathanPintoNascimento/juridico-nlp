import requests, pandas as pd, numpy as np, json, time, os, random
from pathlib import Path
from loguru import logger
from tqdm import tqdm

logger.add("logs/coleta.log", rotation="10 MB", level="INFO")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)
random.seed(42); np.random.seed(42)

def gerar_dados_sinteticos(n_por_classe=300):
    templates = {
        "civil": {
            "intro": ["RECURSO ESPECIAL. DIREITO CIVIL.", "APELACAO CIVEL. ACAO DE INDENIZACAO.", "AGRAVO DE INSTRUMENTO. DIREITO DAS OBRIGACOES."],
            "merito": [
                "Discute-se nos autos a validade de clausula contratual abusiva, com pedido de rescisao do contrato por inadimplemento da parte re. O Tribunal de origem reconheceu a nulidade da clausula, determinando a devolucao em dobro dos valores indevidamente cobrados.",
                "Trata-se de acao de indenizacao por danos morais e materiais decorrentes de acidente de transito. O recorrente alega culpa exclusiva do condutor do veiculo segurado pela recorrida. A pericia tecnica concluiu pela responsabilidade concorrente das partes.",
                "Discute-se a partilha de bens em acao de divorcio litigioso, especificamente a natureza juridica de bem adquirido antes do casamento. O regime adotado pelo casal e o da comunhao parcial de bens, conforme pacto antenupcial registrado em cartorio.",
            ],
            "decisao": ["RECURSO PROVIDO PARCIALMENTE. Afastada a clausula penal excessiva.", "NEGADO PROVIMENTO AO RECURSO. Mantida a sentenca de primeiro grau.", "RECURSO ESPECIAL PROVIDO. Reconhecido o direito a meacao dos bens."],
        },
        "penal": {
            "intro": ["HABEAS CORPUS. DIREITO PENAL. CRIME CONTRA O PATRIMONIO.", "RECURSO ESPECIAL. PENAL. CONDENACAO POR TRAFICO DE DROGAS.", "APELACAO CRIMINAL. CRIME DE HOMICIDIO CULPOSO."],
            "merito": [
                "Trata-se de habeas corpus impetrado em favor do paciente, preso preventivamente por suposta pratica do crime de roubo qualificado pelo emprego de arma de fogo. A defesa alega ausencia dos requisitos legais para decretacao da prisao cautelar, pugnando pela concessao de liberdade provisoria.",
                "Cuida-se de recurso especial interposto pelo Ministerio Publico contra acordao que reduziu a pena aplicada ao reu condenado por trafico ilicito de entorpecentes. O parquet sustenta que a minorante do art. 33 nao e aplicavel em razao da quantidade e variedade de drogas apreendidas.",
                "O reu foi condenado pela pratica de homicidio doloso qualificado por motivo futil. A defesa sustenta que houve violacao do principio da soberania dos veredictos populares, alegando contradicao na resposta dos jurados aos quesitos formulados.",
            ],
            "decisao": ["ORDEM CONCEDIDA. Substituida a prisao preventiva por medidas cautelares.", "RECURSO DESPROVIDO. Mantida a dosimetria da pena.", "HABEAS CORPUS NAO CONHECIDO. Ausencia de ilegalidade flagrante."],
        },
        "trabalhista": {
            "intro": ["RECURSO DE REVISTA. DIREITO DO TRABALHO. RESCISAO INDIRETA.", "AGRAVO DE INSTRUMENTO. HORAS EXTRAS. ADICIONAL NOTURNO.", "RECURSO ORDINARIO. ESTABILIDADE GESTANTE. JUSTA CAUSA."],
            "merito": [
                "Discute-se nos presentes autos a validade da dispensa por justa causa aplicada a empregada gestante. A reclamante alega que o empregador tinha ciencia da gravidez antes da ruptura contratual, fazendo jus a estabilidade provisoria prevista no ADCT.",
                "O reclamante pleiteia o pagamento de horas extras e reflexos, aduzindo que laborava em jornada superior a contratada sem a devida contraprestacao. Os cartoes de ponto juntados aos autos apresentam horarios britanicos, gerando presuncao relativa em favor do trabalhador.",
                "Pedido de rescisao indireta do contrato de trabalho por descumprimento de obrigacoes contratuais pelo empregador, especificamente o nao recolhimento do FGTS por mais de doze meses consecutivos e o atraso reiterado no pagamento de salarios.",
            ],
            "decisao": ["RECURSO PROVIDO. Reconhecida a estabilidade gestante.", "RECURSO PARCIALMENTE PROVIDO. Deferidas as horas extras com adicional de 50%.", "RECURSO DESPROVIDO. Mantido o reconhecimento do vinculo empregaticio."],
        },
        "tributario": {
            "intro": ["RECURSO ESPECIAL. DIREITO TRIBUTARIO. EXECUCAO FISCAL.", "AGRAVO REGIMENTAL. ICMS. SUBSTITUICAO TRIBUTARIA.", "MANDADO DE SEGURANCA. IMPOSTO DE RENDA. ISENCAO."],
            "merito": [
                "Cuida-se de recurso especial em que se discute a legalidade da cobranca de ICMS sobre operacoes de importacao realizadas por pessoa fisica sem habitualidade. A recorrente sustenta que a incidencia do imposto viola o principio da nao-cumulatividade.",
                "Trata-se de mandado de seguranca impetrado contra ato do Secretario de Fazenda que negou pedido de compensacao tributaria. A impetrante alega ter recolhido tributo a maior em decorrencia de equivoco na aplicacao da aliquota.",
                "Execucao fiscal promovida pelo Municipio para cobranca de IPTU progressivo no tempo, em razao de descumprimento da funcao social da propriedade urbana. O executado alega inconstitucionalidade da progressividade antes da EC 29/2000.",
            ],
            "decisao": ["RECURSO PROVIDO. Afastada a incidencia do ICMS na importacao.", "SEGURANCA CONCEDIDA. Reconhecido o direito a compensacao tributaria.", "RECURSO DESPROVIDO. Mantida a tributacao pelo imposto de renda."],
        },
        "administrativo": {
            "intro": ["MANDADO DE SEGURANCA. DIREITO ADMINISTRATIVO. CONCURSO PUBLICO.", "RECURSO ESPECIAL. LICITACAO. MODALIDADE PREGAO.", "ACAO POPULAR. ATO ADMINISTRATIVO. ANULACAO."],
            "merito": [
                "Mandado de seguranca impetrado por candidato aprovado em concurso publico dentro do numero de vagas, visando a nomeacao. A autoridade coatora alega inexistencia de direito liquido e certo, por tratar-se de ato discricionario da Administracao Publica.",
                "Trata-se de impugnacao a edital de licitacao na modalidade pregao eletronico para aquisicao de equipamentos de informatica. O recorrente sustenta que as especificacoes tecnicas do edital sao direcionadas a marca especifica, violando o principio da competitividade.",
                "Acao de improbidade administrativa movida pelo Ministerio Publico em face de ex-gestor municipal. Imputa-se ao reu o desvio de verbas publicas destinadas a saude, mediante contratacao de empresa fantasma com dispensa de licitacao.",
            ],
            "decisao": ["SEGURANCA CONCEDIDA. Reconhecido o direito subjetivo a nomeacao.", "RECURSO PROVIDO. Declarada a nulidade da licitacao.", "RECURSO DESPROVIDO. Mantida a condenacao por improbidade administrativa."],
        },
        "consumidor": {
            "intro": ["RECURSO ESPECIAL. DIREITO DO CONSUMIDOR. DANO MORAL.", "APELACAO CIVEL. RELACAO DE CONSUMO. PRODUTO DEFEITUOSO.", "AGRAVO DE INSTRUMENTO. CDC. INVERSAO DO ONUS DA PROVA."],
            "merito": [
                "Discute-se nos autos se a negativacao indevida do nome do consumidor em cadastros restritivos de credito configura dano moral in re ipsa, prescindindo de comprovacao do prejuizo. O recorrente foi inscrito no SPC e SERASA em razao de debito ja quitado.",
                "Acao de indenizacao por danos materiais e morais decorrentes de acidente de consumo causado por produto defeituoso. A recorrida, fabricante do produto, alega culpa exclusiva do consumidor pelo uso em desacordo com as instrucoes do fabricante.",
                "Pedido de declaracao de nulidade de clausula contratual de plano de saude que limita o periodo de internacao hospitalar. O consumidor sustenta que a restricao e abusiva e contraria o art. 51 do Codigo de Defesa do Consumidor.",
            ],
            "decisao": ["RECURSO PROVIDO. Reconhecida a responsabilidade objetiva do fornecedor.", "RECURSO DESPROVIDO. Mantida a condenacao por danos morais.", "RECURSO PARCIALMENTE PROVIDO. Reduzido o quantum indenizatorio."],
        },
    }
    registros = []
    logger.info(f"Gerando {n_por_classe} amostras por classe...")
    for classe, c in tqdm(templates.items(), desc="Classes"):
        for i in range(n_por_classe):
            texto = f"{random.choice(c['intro'])}\n\n{random.choice(c['merito'])}\n\n{random.choice(c['decisao'])}"
            registros.append({"texto": texto, "classe": classe, "fonte": "sintetico_realista",
                               "tribunal": random.choice(["STJ","TJSP","TJRJ","TJMG","TRT"]),
                               "numero": f"{random.randint(10000,9999999)}-{random.randint(10,99)}.{random.randint(2015,2024)}.1.{random.randint(10,99)}.{random.randint(1000,9999)}"})
    df = pd.DataFrame(registros)
    logger.success(f"Geradas {len(df)} amostras sinteticas")
    return df

def salvar_dados(df, nome="dataset_juridico"):
    df.to_csv(DATA_RAW / f"{nome}.csv", index=False, encoding="utf-8-sig")
    df.to_parquet(DATA_RAW / f"{nome}.parquet", index=False)
    meta = {"total": len(df), "classes": df["classe"].value_counts().to_dict()}
    with open(DATA_RAW / f"{nome}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.success(f"Dados salvos em {DATA_RAW}")
    return DATA_RAW / f"{nome}.csv", DATA_RAW / f"{nome}.parquet"

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    print("\n" + "="*60 + "\n  ETAPA 1 - COLETA DE DADOS JURIDICOS\n" + "="*60)
    df = gerar_dados_sinteticos(n_por_classe=300)
    csv_path, parquet_path = salvar_dados(df)
    print(f"\nTotal: {len(df)} documentos em {df['classe'].nunique()} classes")
    print(f"CSV:     {csv_path}")
    print(f"Parquet: {parquet_path}")
    print("\nPROXIMA ETAPA: python src/eda.py\n")
