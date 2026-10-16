"""Gera o recorte 1924-1935 usado no artigo a partir do dataset público do Kaggle.

1. Baixe o arquivo Market_Trend_External.csv em
   https://www.kaggle.com/datasets/kundanbedmutha/market-trend-and-external-factors-dataset
2. Coloque-o na pasta dados/
3. Rode:  python preparar_recorte.py
"""
import hashlib
import pandas as pd

ORIGEM = 'dados/Market_Trend_External.csv'
SAIDA = 'dados/recorte_1924_1935.csv'

df = pd.read_csv(ORIGEM, parse_dates=['Date'])
recorte = df[(df['Date'] >= '1924-01-01') & (df['Date'] <= '1935-12-31')]
recorte.to_csv(SAIDA, index=False)

print(f'Observações no recorte: {len(recorte)} (esperado: 3131)')
print(f'Período: {recorte.Date.min().date()} a {recorte.Date.max().date()}')
print('SHA-256 do arquivo original:', hashlib.sha256(open(ORIGEM, 'rb').read()).hexdigest())
print('(esperado: ce02fcbca31f817f901288dcdfcef3d45ac331fb9eecad1bf77e37f9f7cdff13)')
