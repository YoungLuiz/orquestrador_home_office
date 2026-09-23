"""
Orquestrador de escala presencial — simulador interativo.

Rodar com:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.avaliador import avaliar, comparar, matriz_carga
from src.dados import DIAS, escala_baseline, gerar_cenario
from src.otimizador import Pesos, otimizar

st.set_page_config(page_title="Orquestrador de escala presencial", layout="wide")

ROTULO = {"seg": "Segunda", "ter": "Terça", "qua": "Quarta", "qui": "Quinta", "sex": "Sexta"}


# --------------------------------------------------------------------------
# Controles
# --------------------------------------------------------------------------
st.sidebar.header("Cluster")
n_emp = st.sidebar.slider("Empresas no cluster", 20, 150, 90, step=10)
seed = st.sidebar.number_input("Semente aleatória", value=42, step=1)

st.sidebar.header("Política de presença")
fator_postos = st.sidebar.slider(
    "Postos de trabalho disponíveis (% do headcount)", 0.75, 1.00, 1.00, step=0.05
)
dia_comum = st.sidebar.checkbox(
    "Exigir 1 dia com a empresa inteira", value=True,
    help="Garante all-hands. Pode tornar o problema inviável se os postos forem poucos.",
)

st.sidebar.header("Pesos do objetivo")
w_pico = st.sidebar.slider("Achatar o pico", 0, 300, 100, step=10)
w_exc = st.sidebar.slider("Punir excesso de capacidade", 0, 20, 1)
w_pref = st.sidebar.slider("Respeitar preferência dos times", 0, 50, 5)
limite_s = st.sidebar.slider("Tempo máximo do solver (s)", 5, 60, 20, step=5)


@st.cache_data(show_spinner=False)
def montar(n: int, s: int):
    c = gerar_cenario(n_empresas=n, seed=s)
    return c, escala_baseline(c, seed=s + 1)


@st.cache_data(show_spinner=False)
def resolver(n, s, fp, dc, wp, we, wf, lim):
    c, _ = montar(n, s)
    return otimizar(
        c,
        pesos=Pesos(pico=wp, excesso=we, preferencia=wf),
        exigir_dia_comum=dc,
        fator_postos=fp,
        limite_s=lim,
    )


cenario, base = montar(n_emp, seed)

st.title("Orquestrador de escala presencial")
st.caption(
    f"{len(cenario.empresas)} empresas · {len(cenario.times)} times · "
    f"{cenario.total_pessoas:,} colaboradores · 8 corredores de chegada".replace(",", ".")
)

with st.spinner("Resolvendo o modelo..."):
    res = resolver(n_emp, seed, fator_postos, dia_comum, w_pico, w_exc, w_pref, limite_s)

if not res.viavel:
    st.error(f"Modelo inviável ({res.status}).\n\n{res.gap_info}")
    st.stop()

m_base = avaliar(cenario, base)
m_otim = avaliar(cenario, res.escala)
delta = comparar(m_base, m_otim)

# --------------------------------------------------------------------------
# Indicadores
# --------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Pico de ocupação",
    f"{m_otim.pico_pct:.0f}%",
    f"{delta['pico_pct']:.1f}%",
    delta_color="inverse",
    help=f"Antes: {m_base.pico_pct:.0f}% em {m_base.recurso_pico} na {ROTULO[m_base.dia_pico].lower()}",
)
c2.metric(
    "Gargalos (corredor × dia acima de 100%)",
    m_otim.celulas_acima,
    m_otim.celulas_acima - m_base.celulas_acima,
    delta_color="inverse",
)
c3.metric(
    "Pessoas além da capacidade",
    f"{m_otim.pessoas_exc:,}".replace(",", "."),
    f"{delta['pessoas_exc']:.0f}%",
    delta_color="inverse",
)
c4.metric(
    "Desequilíbrio entre dias",
    f"{m_otim.desvio_dias:,.0f}".replace(",", "."),
    f"{delta['desvio_dias']:.0f}%",
    delta_color="inverse",
    help="Desvio padrão da carga total entre os cinco dias. Quanto menor, mais plana a semana.",
)

st.caption(f"Solver: {res.status} em {res.tempo_s:.1f}s — {res.gap_info}")

if res.sem_posto:
    st.warning(
        f"{res.sem_posto:,} pessoas-dia ficariam sem estação de trabalho nesta configuração. "
        "O escritório virou o gargalo, não o transporte. "
        "Aumente o percentual de postos ou desmarque o dia com a empresa inteira."
        .replace(",", ".")
    )

# --------------------------------------------------------------------------
# Carga por dia
# --------------------------------------------------------------------------
st.subheader("Pessoas chegando ao cluster por dia")

df_dia = pd.DataFrame(
    {
        "Dia": [ROTULO[d] for d in DIAS] * 2,
        "Pessoas": [m_base.carga_por_dia[d] for d in DIAS]
        + [m_otim.carga_por_dia[d] for d in DIAS],
        "Cenário": ["Atual"] * 5 + ["Otimizado"] * 5,
    }
)
fig = px.bar(
    df_dia, x="Dia", y="Pessoas", color="Cenário", barmode="group",
    color_discrete_map={"Atual": "#c1443f", "Otimizado": "#2d7d64"},
)
fig.update_layout(height=380, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Mapa de calor por corredor
# --------------------------------------------------------------------------
st.subheader("Ocupação por corredor de chegada (% da capacidade)")

def heat(escala, titulo):
    mat = matriz_carga(cenario, escala)
    recursos = list(cenario.recursos)
    z = [[100 * mat[r][d] / cenario.recursos[r] for d in DIAS] for r in recursos]
    f = go.Figure(
        go.Heatmap(
            z=z,
            x=[ROTULO[d] for d in DIAS],
            y=recursos,
            colorscale=[[0, "#f2f7f4"], [0.55, "#8fbfa9"], [0.8, "#e8c46a"], [1, "#b02f28"]],
            zmin=0, zmax=140,
            texttemplate="%{z:.0f}%",
            textfont={"size": 11},
            colorbar=dict(title="%"),
        )
    )
    f.update_layout(title=titulo, height=420, margin=dict(t=50, b=20))
    return f

h1, h2 = st.columns(2)
h1.plotly_chart(heat(base, "Cenário atual"), use_container_width=True)
h2.plotly_chart(heat(res.escala, "Cenário otimizado"), use_container_width=True)

# --------------------------------------------------------------------------
# Escala resultante
# --------------------------------------------------------------------------
st.subheader("Escala sugerida")

emp_sel = st.selectbox(
    "Empresa", [e.id for e in cenario.empresas],
    format_func=lambda i: next(e.nome for e in cenario.empresas if e.id == i),
)
pol = next(e for e in cenario.empresas if e.id == emp_sel)
st.caption(f"Política: {pol.dias_min} a {pol.dias_max} dias presenciais por semana")

linhas = []
for t in cenario.times_da(emp_sel):
    linha = {"Time": t.id.split("-")[-1], "Pessoas": t.headcount}
    for d in DIAS:
        linha[ROTULO[d]] = "●" if d in res.escala[t.id] else ""
    linha["Âncora"] = ROTULO.get(t.dia_ancora, "—")
    linhas.append(linha)
st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

csv = pd.DataFrame(
    [
        {"time": t.id, "empresa": t.empresa, "headcount": t.headcount,
         **{d: int(d in res.escala[t.id]) for d in DIAS}}
        for t in cenario.times
    ]
).to_csv(index=False).encode()
st.download_button("Baixar escala completa (CSV)", csv, "escala_otimizada.csv", "text/csv")
