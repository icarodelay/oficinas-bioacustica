# Oficinas de Bioacústica: Machine Learning e Processamento de Áudio de Voz

Bem-vindo às oficinas de bioacústica! 

A motivação para este conteúdo é para repasse técnico do nosso grupo de estudos, porém sinta-se convidado a estudar mesmo assim.
Este material introduz técnicas de Machine Learning e processamento de sinais de áudio com foco em análise de voz humana, usando Python e Jupyter Notebooks.

✉️ `icarodelima@alu.ufc.br`

---

## Estrutura

```
oficinas-bioacustica/
│
├── modulo_01/   — Padrões de Projeto ML em Python
│   ├── 01_python_para_ml.ipynb
│   ├── 02_padroes_projeto_ml.ipynb
│   └── 03_sklearn_fundamentos.ipynb
│
├── modulo_02/   — Técnicas de ML, Avaliação e Visualização
│   ├── 01_algoritmos_scratch.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_metricas_avaliacao.ipynb
│
├── modulo_03/   — Processamento de Áudio e Voz
│   ├── 01_fundamentos_audio.ipynb
│   ├── 02_processamento_sinais.ipynb
│   ├── 03_extracao_features_voz.ipynb
│   └── 04_filtros_aumentacao.ipynb
│
└── modulo_04/   — Projeto Final Ponta-a-Ponta
    └── 01_projeto_final.ipynb
```

---

## Módulos

### Módulo 01 — Padrões de Projeto ML em Python
Introdução às bibliotecas essenciais (NumPy, Pandas, Matplotlib) e aos principais padrões de projeto aplicados a sistemas de Machine Learning: Pipeline, Strategy, Factory e Observer.

### Módulo 02 — Machine Learning em Python
Implementação de algoritmos clássicos from scratch (KNN, Naive Bayes, Regressão Logística), engenharia de features, redução de dimensionalidade e avaliação rigorosa de modelos com métricas, validação cruzada e curvas de aprendizado.

### Módulo 03 — Processamento de Áudio e Voz
Fundamentos de áudio digital, Transformada de Fourier, espectrogramas, extração de features de voz (MFCC, Chroma, pitch), filtros de sinal e técnicas de aumento de dados para áudio.

### Módulo 04 — Projeto Final
Pipeline completo de reconhecimento de emoção em voz: coleta de dados, pré-processamento, extração de features, treinamento de modelos clássicos e de redes neurais, avaliação e análise de erros.

---

## Instalação

```bash
# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt

# Inicie o Jupyter
jupyter notebook
```

## Pré-requisitos

- Python 3.9+
- Conhecimento básico de programação Python
- Noções de álgebra linear e estatística (útil, não obrigatório)

---

## Datasets Utilizados

- **RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- **CREMA-D** — Crowd-sourced Emotional Multimodal Actors Dataset  
- Exemplos sintéticos gerados nos próprios notebooks

---

Desenvolvido para as Oficinas de Bioacústica — 2026
