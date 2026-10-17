# Backtest walk-forward — série simulada, 1924-1935

Código e instruções de reprodução do artigo **"Análise de padrões estatísticos em dados simulados do mercado acionário americano no período da Grande Depressão (1924-1935)"** (Santiago, Marassi, Zumba e Azevedo — IX Semana de Ciências e Tecnologia, UFRN, 2026).

## Conteúdo

| Arquivo | O que faz |
|---|---|
| `preparar_recorte.py` | Gera `dados/recorte_1924_1935.csv` (3.131 dias) a partir do dataset público |
| `backtest_conferencia.py` | Script de conferência dos autores. Gera as Tabelas 2 e 3 e as Figuras 1 e 2 do artigo em `resultados/` |
| `notebook/BACK_MODULAR_QUANT_artigo1.ipynb` | Notebook BACK_MODULAR_QUANT (grupo QuantInovarum / disciplina de Finanças Quantitativas), com a célula 1 adaptada para ler o dataset |
| `requirements.txt` | Dependências (pandas, NumPy, Matplotlib) |

## Resultados prontos

A pasta `resultados/` já traz as saídas do script, para consulta sem precisar rodar nada: `tabela2.csv` e `tabela3.csv` (Tabelas 2 e 3 do artigo, com a coluna extra `Vol. B&H`, volatilidade anualizada do buy-and-hold citada no texto) e `figura1.png` e `figura2.png` (idênticas às Figuras 1 e 2 do artigo).

## Dados

O dataset **não** é redistribuído aqui. Baixe `Market_Trend_External.csv` em
<https://www.kaggle.com/datasets/kundanbedmutha/market-trend-and-external-factors-dataset>
(BEDMUTHA, 2025) e coloque-o em `dados/`. A série é sintética: não reproduz a trajetória real dos índices americanos.

SHA-256 do arquivo usado no artigo: `ce02fcbca31f817f901288dcdfcef3d45ac331fb9eecad1bf77e37f9f7cdff13`

## Como reproduzir

```bash
pip install -r requirements.txt
python preparar_recorte.py        # cria dados/recorte_1924_1935.csv
python backtest_conferencia.py    # imprime as tabelas e salva figuras em resultados/
```

Para rodar o notebook, abra `notebook/BACK_MODULAR_QUANT_artigo1.ipynb` e execute as células de cima para baixo. A célula 1 lê os dados de 1924 a 1936 porque o laço do notebook descarta o último ano carregado; assim o último ano de teste é 1935.

## Procedimento

- Regra: comprado quando a média móvel curta (3 a 15 dias) está acima da longa (30 a 70 dias); fora do mercado no caso contrário. A regra é apenas comprada (533 combinações).
- Custo: 0,3% em cada entrada ou saída. Sem stop loss ou stop gain.
- Walk-forward: 5 anos de treino e 1 ano de teste; no treino escolhe-se a combinação de maior Sharpe, aplicada sem reajuste ao ano seguinte (primeiro teste: 1929; último: 1935).
- O sinal de cada dia vale para o retorno do dia seguinte (sem look-ahead). Anualização com 252 dias.

## Diferença entre o notebook e o script

As duas implementações escolhem os mesmos parâmetros em todos os anos. O script lança o custo de transação no dia do sinal, um dia antes do notebook; por isso os Sharpes diferem em até 0,02 e os retornos, drawdowns e volatilidades em até 0,2 ponto percentual. Os números publicados no artigo (Tabelas 2 e 3, Figuras 1 e 2) são os do script.
