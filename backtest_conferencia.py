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
        'CAGR': eq.iloc[-1] ** (252 / len(eq)) - 1, 'Vol.': est.std() * np.sqrt(252), 'Vol. B&H': bh.std() * np.sqrt(252),
        'Trades': int((sinal.diff().abs() == 1).sum()),
        'Dias compr. (%)': 100 * sinal.shift(1).fillna(0).mean(), 'Parâm.': f'MA{c}/{l}'})
    serie_estr += list(est); serie_bh += list(bh); datas += list(teste.Date)

t2 = pd.DataFrame(linhas)
fases = {'Expansão / Crash': [1929], 'Contração': [1930, 1931, 1932, 1933], 'Recuperação': [1934, 1935]}
t3 = pd.DataFrame([{'Fase': f, 'Sharpe médio': t2[t2.Ano.isin(a)].Sharpe.mean(),
                    'Vol. média': t2[t2.Ano.isin(a)]['Vol.'].mean(), 'DD médio': t2[t2.Ano.isin(a)]['DD máx.'].mean(),
                    'Trades médios': t2[t2.Ano.isin(a)].Trades.mean(), 'Vol. B&H média': t2[t2.Ano.isin(a)]['Vol. B&H'].mean(),
                    'Dias compr. (%)': t2[t2.Ano.isin(a)]['Dias compr. (%)'].mean()} for f, a in fases.items()])
t2.to_csv('resultados/tabela2.csv', index=False); t3.to_csv('resultados/tabela3.csv', index=False)
pd.set_option('display.width', 200)
print(t2.round(4).to_string(index=False)); print(); print(t3.round(4).to_string(index=False))
print('\nMédia diária OOS (todos os dias):', np.mean(serie_estr))

# ---------------- figuras (mesmo estilo das Figuras 1 e 2 do artigo) ----------------
from matplotlib.lines import Line2D
STRAT, BH, INK, MUTED, GRID = '#2A62A8', '#B9781A', '#222222', '#666666', '#DDDDDD'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
recs = [dict(year=int(r['Ano']), sharpe=r['Sharpe'], sharpe_bh=r['Sharpe B&H'], ret=r['Ret. estr.'], ret_bh=r['Ret. B&H'],
             dd=r['DD máx.'], cagr=r['CAGR'], vol=r['Vol.'], trades=r['Trades']) for _, r in t2.iterrows()]
years = [r['year'] for r in recs]; x = np.arange(len(years))
dates, st_all = datas, serie_estr
eq_s, eq_b = np.cumprod(1 + np.array(serie_estr)), np.cumprod(1 + np.array(serie_bh))
d = df
pct = plt.FuncFormatter(lambda v, _: f'{v:.0%}'.replace('.', ','))
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
def yearticks(a):
    a.set_xticks(x); a.set_xticklabels([str(y) for y in years], fontsize=6.3)
def style(a):
    a.grid(axis='y', color=GRID, linewidth=0.5); a.set_axisbelow(True)

# ---------------- Figura 1: a-d ----------------
fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.5)); ax = axs.flatten()
for a in ax: style(a)
a = ax[0]
a.plot(dates, eq_s, color=STRAT, lw=1.2); a.plot(dates, eq_b, color=BH, lw=1.2)
a.axhline(1, color=MUTED, lw=0.5, ls=':')
a.xaxis.set_major_locator(mdates.YearLocator(1)); a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
a.tick_params(axis='x', labelsize=6.3)
a.set_title('(a) Capital acumulado OOS (base = 1)')
a = ax[1]
a.bar(x, [r['sharpe'] for r in recs], color=STRAT, width=0.6)
a.plot(x, [r['sharpe_bh'] for r in recs], 'o', color=BH, ms=4)
a.axhline(0, color=INK, lw=0.6); yearticks(a); a.set_title('(b) Sharpe Ratio anual')
a = ax[2]; w = 0.38
a.bar(x - w/2, [r['ret'] for r in recs], w, color=STRAT); a.bar(x + w/2, [r['ret_bh'] for r in recs], w, color=BH)
a.axhline(0, color=INK, lw=0.6); a.yaxis.set_major_formatter(pct); yearticks(a); a.set_title('(c) Retorno anual')
a = ax[3]
a.bar(x, [r['dd'] for r in recs], color=STRAT, width=0.6)
a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05)); a.yaxis.set_major_formatter(pct); yearticks(a); a.set_title('(d) Drawdown máximo da estratégia')
fig.legend(handles=[Patch(color=STRAT), Patch(color=BH)],
           labels=['Estratégia (cruzamento de médias)', 'Buy-and-hold (barras em c; pontos em b)'],
           loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.4, h_pad=0.9, w_pad=0.8, rect=(0, 0, 1, 0.95))
fig.savefig('resultados/figura1.png', dpi=300); plt.close(fig)

# ---------------- Figura 2: e-i ----------------
fig = plt.figure(figsize=(6.3, 4.4))
gs = fig.add_gridspec(2, 3)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
      fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])]
for a in ax: style(a)
a = ax[0]
a.plot(x, [r['vol'] for r in recs], '-o', color=STRAT, lw=1.2, ms=3.5)
a.set_ylim(0, 0.18); a.yaxis.set_major_formatter(pct); yearticks(a); a.set_title('(e) Volatilidade anualizada')
a = ax[1]
a.bar(x, [r['cagr'] for r in recs], color=STRAT, width=0.6)
a.axhline(0, color=INK, lw=0.6); a.yaxis.set_major_formatter(pct); yearticks(a); a.set_title('(f) CAGR anual')
a = ax[2]
a.bar(x, [r['trades'] for r in recs], color=STRAT, width=0.6)
yearticks(a); a.set_title('(g) Número de trades'); a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
a = ax[3]
arr = np.array(st_all); arr = arr[arr != 0]
a.hist(arr, bins=35, color=STRAT, edgecolor='white', linewidth=0.3)
m = np.mean(st_all)
a.axvline(m, color=BH, lw=1.2, ls='--')
a.text(0.98, 0.95, f'média = {m:.5f}'.replace('.', ','), transform=a.transAxes, ha='right', va='top', color=INK, fontsize=6.3)
a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'.replace('.', ',')))
a.set_title('(h) Retornos diários OOS (dias comprado)'); a.set_ylabel('Frequência')
a = ax[4]
a.plot(d.Date, d.Close, color='#AAAAAA', lw=0.6, label='Preço simulado')
a.plot(d.Date, d.Close.rolling(8).mean(), color=STRAT, lw=0.9, label='MM 8 dias')
a.plot(d.Date, d.Close.rolling(33).mean(), color=BH, lw=0.9, label='MM 33 dias')
a.xaxis.set_major_locator(mdates.YearLocator(1)); a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
a.tick_params(axis='x', labelsize=6.3)
a.set_title('(i) Preço simulado e médias móveis'); a.legend(frameon=False, loc='upper left')
fig.tight_layout(pad=0.4, h_pad=0.9, w_pad=0.8)
fig.savefig('resultados/figura2.png', dpi=300); plt.close(fig)
print('Tabelas e figuras salvas em resultados/')
