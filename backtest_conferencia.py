"""Script de conferência dos autores: reproduz as Tabelas 2 e 3 e as Figuras 1 e 2 do artigo
"Análise de padrões estatísticos em dados simulados do mercado acionário americano no período
da Grande Depressão (1924-1935)".

Mesmo procedimento do notebook BACK_MODULAR_QUANT (grade 3-15 x 30-70, custo de 0,3% por
entrada ou saída, treino de 5 anos e teste de 1 ano, escolha pelo maior Sharpe no treino).
Única diferença: o custo é lançado no dia do sinal (um dia antes do notebook), o que muda os
Sharpes em até 0,02 e retornos/drawdowns/volatilidades em até 0,2 ponto percentual.

Uso:  python backtest_conferencia.py   (depois de rodar preparar_recorte.py)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DADOS = 'dados/recorte_1924_1935.csv'
CUSTO = 0.003
TREINO, ANO_INI, ANO_FIM = 5, 1924, 1935
GRADE = [(c, l) for c in range(3, 16) for l in range(30, 71)]          # 533 combinações

df = pd.read_csv(DADOS, parse_dates=['Date']).rename(columns={'Close_Price': 'Close'})
df = df[(df.Date.dt.year >= ANO_INI) & (df.Date.dt.year <= ANO_FIM)].reset_index(drop=True)
df['Year'] = df.Date.dt.year


def sharpe(r):
    s = r.std()
    return r.mean() / s * np.sqrt(252) if s else 0.0          # anualização com 252 dias


def retornos(janela, curta, longa):
    close = janela['Close']
    sinal = pd.Series(np.where(close.rolling(curta).mean() > close.rolling(longa).mean(), 1, 0), index=janela.index)
    r_dia = close.pct_change().fillna(0)
    trocas = sinal.diff().abs().fillna(0)
    estrategia = sinal.shift(1).fillna(0) * r_dia - trocas * CUSTO   # sinal de hoje vale para amanhã
    return sinal, estrategia, r_dia


anos = sorted(df.Year.unique())
linhas, serie_estr, serie_bh, datas = [], [], [], []
for i in range(len(anos) - TREINO):
    treino = df[df.Year.between(anos[i], anos[i + TREINO - 1])]
    teste = df[df.Year == anos[i + TREINO]]
    c, l = max(GRADE, key=lambda p: sharpe(retornos(treino, *p)[1]))
    sinal, est, bh = retornos(teste, c, l)
    eq = (1 + est).cumprod(); eq_bh = (1 + bh).cumprod()
    linhas.append({
        'Ano': anos[i + TREINO], 'Sharpe': sharpe(est), 'Sharpe B&H': sharpe(bh),
        'Ret. estr.': eq.iloc[-1] - 1, 'Ret. B&H': eq_bh.iloc[-1] - 1,
        'DD máx.': ((eq - eq.cummax()) / eq.cummax()).min(),
        'CAGR': eq.iloc[-1] ** (252 / len(eq)) - 1, 'Vol.': est.std() * np.sqrt(252),
        'Trades': int((sinal.diff().abs() == 1).sum()),
        'Dias compr. (%)': 100 * sinal.shift(1).fillna(0).mean(), 'Parâm.': f'MA{c}/{l}'})
    serie_estr += list(est); serie_bh += list(bh); datas += list(teste.Date)

t2 = pd.DataFrame(linhas)
fases = {'Expansão / Crash': [1929], 'Contração': [1930, 1931, 1932, 1933], 'Recuperação': [1934, 1935]}
t3 = pd.DataFrame([{'Fase': f, 'Sharpe médio': t2[t2.Ano.isin(a)].Sharpe.mean(),
                    'Vol. média': t2[t2.Ano.isin(a)]['Vol.'].mean(), 'DD médio': t2[t2.Ano.isin(a)]['DD máx.'].mean(),
                    'Trades médios': t2[t2.Ano.isin(a)].Trades.mean(),
                    'Dias compr. (%)': t2[t2.Ano.isin(a)]['Dias compr. (%)'].mean()} for f, a in fases.items()])
t2.to_csv('resultados/tabela2.csv', index=False); t3.to_csv('resultados/tabela3.csv', index=False)
pd.set_option('display.width', 200)
print(t2.round(4).to_string(index=False)); print(); print(t3.round(4).to_string(index=False))
print('\nMédia diária OOS (todos os dias):', np.mean(serie_estr))

# ---------------- figuras ----------------
AZUL, OCRE, CINZA = '#2A62A8', '#B9781A', '#DDDDDD'
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
x = np.arange(len(t2)); pct = plt.FuncFormatter(lambda v, _: f'{v:.0%}'.replace('.', ','))
eq_s, eq_b = np.cumprod(1 + np.array(serie_estr)), np.cumprod(1 + np.array(serie_bh))

def anos_x(a):
    a.set_xticks(x); a.set_xticklabels(t2.Ano.astype(str)); a.grid(axis='y', color=CINZA, lw=0.5); a.set_axisbelow(True)

fig, ax = plt.subplots(2, 2, figsize=(6.3, 4.5)); ax = ax.flatten()
ax[0].plot(datas, eq_s, color=AZUL); ax[0].plot(datas, eq_b, color=OCRE); ax[0].axhline(1, ls=':', lw=0.5, color='gray')
ax[0].set_title('(a) Capital acumulado OOS (base = 1)')
ax[1].bar(x, t2.Sharpe, color=AZUL, width=0.6); ax[1].plot(x, t2['Sharpe B&H'], 'o', color=OCRE); ax[1].axhline(0, color='k', lw=0.6)
anos_x(ax[1]); ax[1].set_title('(b) Sharpe anual (pontos: buy-and-hold)')
ax[2].bar(x - 0.19, t2['Ret. estr.'], 0.38, color=AZUL); ax[2].bar(x + 0.19, t2['Ret. B&H'], 0.38, color=OCRE)
ax[2].axhline(0, color='k', lw=0.6); ax[2].yaxis.set_major_formatter(pct); anos_x(ax[2]); ax[2].set_title('(c) Retorno anual')
ax[3].bar(x, t2['DD máx.'], color=AZUL, width=0.6); ax[3].yaxis.set_major_formatter(pct); anos_x(ax[3])
ax[3].set_title('(d) Drawdown máximo da estratégia')
fig.tight_layout(); fig.savefig('resultados/figura1.png', dpi=300); plt.close(fig)

fig = plt.figure(figsize=(6.3, 4.4)); gs = fig.add_gridspec(2, 3)
a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])]
a[0].plot(x, t2['Vol.'], '-o', color=AZUL); a[0].yaxis.set_major_formatter(pct); anos_x(a[0]); a[0].set_title('(e) Volatilidade anualizada')
a[1].bar(x, t2.CAGR, color=AZUL, width=0.6); a[1].yaxis.set_major_formatter(pct); anos_x(a[1]); a[1].set_title('(f) CAGR anual')
a[2].bar(x, t2.Trades, color=AZUL, width=0.6); anos_x(a[2]); a[2].set_title('(g) Número de trades')
arr = np.array(serie_estr); a[3].hist(arr[arr != 0], bins=35, color=AZUL); a[3].axvline(arr.mean(), color=OCRE, ls='--')
a[3].set_title('(h) Retornos diários (dias comprado)')
a[4].plot(df.Date, df.Close, color='#AAAAAA', lw=0.6, label='Preço simulado')
a[4].plot(df.Date, df.Close.rolling(8).mean(), color=AZUL, lw=0.9, label='MM 8 dias')
a[4].plot(df.Date, df.Close.rolling(33).mean(), color=OCRE, lw=0.9, label='MM 33 dias')
a[4].legend(frameon=False); a[4].set_title('(i) Preço simulado e médias móveis')
fig.tight_layout(); fig.savefig('resultados/figura2.png', dpi=300); plt.close(fig)
print('Tabelas e figuras salvas em resultados/')
