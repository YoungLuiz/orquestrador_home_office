"""
Geracao da populacao sintetica de empresas, times e colaboradores.

O recorte e um cluster corporativo (ex.: Faria Lima / Itaim). Cada colaborador
chega a regiao por exatamente um RECURSO (linha de metro, corredor viario,
ponte). A unidade de decisao do otimizador nao e a pessoa, e o TIME: pessoas do
mesmo time precisam estar juntas, entao modelar por time reduz o problema em
uma ordem de grandeza e ja embute a restricao de co-presenca.

IMPORTANTE PARA O TCC: as capacidades abaixo sao placeholders plausiveis, nao
dados oficiais. O passo de calibracao consiste em substitui-las por numeros
derivados da Pesquisa Origem e Destino do Metro de SP e dos boletins da CET.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

DIAS = ["seg", "ter", "qua", "qui", "sex"]

# Capacidade confortavel de chegada na regiao no pico da manha (pessoas/hora).
# Trocar por dados reais na etapa de calibracao.
RECURSOS: dict[str, int] = {
    "L4-Amarela": 4200,
    "L9-Esmeralda": 3400,
    "L5-Lilas": 2600,
    "CPTM-L8": 1900,
    "Av-Faria-Lima": 1700,
    "Marginal-Pinheiros": 2500,
    "Ponte-Cidade-Jardim": 1100,
    "Corredor-Onibus": 1400,
}

# Peso de atracao de cada recurso para o cluster. A Linha 4 serve diretamente a
# regiao, entao concentra desproporcionalmente a demanda. E justamente essa
# assimetria que cria o gargalo.
ATRACAO: dict[str, float] = {
    "L4-Amarela": 0.26,
    "L9-Esmeralda": 0.17,
    "L5-Lilas": 0.10,
    "CPTM-L8": 0.07,
    "Av-Faria-Lima": 0.12,
    "Marginal-Pinheiros": 0.15,
    "Ponte-Cidade-Jardim": 0.06,
    "Corredor-Onibus": 0.07,
}

# Distribuicao observada de escolha de dia presencial no cenario atual.
# Reproduz o padrao ter/qua/qui documentado na literatura de trabalho hibrido.
PESO_DIA_BASELINE = [0.10, 0.28, 0.32, 0.25, 0.05]


@dataclass
class Time:
    id: str
    empresa: str
    headcount: int
    por_recurso: dict[str, int]
    dias_preferidos: list[str] = field(default_factory=list)
    dia_ancora: str | None = None


@dataclass
class Empresa:
    id: str
    nome: str
    dias_min: int
    dias_max: int
    postos: int  # estacoes de trabalho disponiveis por dia


@dataclass
class Cenario:
    empresas: list[Empresa]
    times: list[Time]
    recursos: dict[str, int]

    def times_da(self, empresa_id: str) -> list[Time]:
        return [t for t in self.times if t.empresa == empresa_id]

    @property
    def total_pessoas(self) -> int:
        return sum(t.headcount for t in self.times)


def _sortear_recursos(headcount: int, rng: random.Random) -> dict[str, int]:
    """Distribui as pessoas de um time entre os recursos de chegada."""
    nomes = list(ATRACAO)
    pesos = [ATRACAO[n] for n in nomes]
    escolhas = rng.choices(nomes, weights=pesos, k=headcount)
    dist: dict[str, int] = {}
    for r in escolhas:
        dist[r] = dist.get(r, 0) + 1
    return dist


def gerar_cenario(
    n_empresas: int = 90,
    times_por_empresa: tuple[int, int] = (4, 9),
    tamanho_time: tuple[int, int] = (18, 60),
    seed: int = 42,
) -> Cenario:
    rng = random.Random(seed)
    empresas: list[Empresa] = []
    times: list[Time] = []

    for i in range(n_empresas):
        eid = f"E{i:03d}"
        dias_min = rng.choice([2, 2, 3, 3, 3, 4])
        dias_max = min(5, dias_min + rng.choice([0, 1, 1]))

        n_times = rng.randint(*times_por_empresa)
        headcounts = [rng.randint(*tamanho_time) for _ in range(n_times)]
        total = sum(headcounts)

        empresas.append(
            Empresa(
                id=eid,
                nome=f"Empresa {i:03d}",
                dias_min=dias_min,
                dias_max=dias_max,
                # Escritorio dimensionado para caber todo mundo no dia comum.
                # O slider de ocupacao no app reduz isso para testar pressao.
                postos=total,
            )
        )

        for j, hc in enumerate(headcounts):
            # Alguns times tem ritual fixo (all-hands, comite, fechamento).
            ancora = rng.choice(DIAS) if rng.random() < 0.18 else None
            prefs = rng.sample(DIAS, k=min(dias_min, 5))
            times.append(
                Time(
                    id=f"{eid}-T{j}",
                    empresa=eid,
                    headcount=hc,
                    por_recurso=_sortear_recursos(hc, rng),
                    dias_preferidos=prefs,
                    dia_ancora=ancora,
                )
            )

    return Cenario(empresas=empresas, times=times, recursos=dict(RECURSOS))


def escala_baseline(cenario: Cenario, seed: int = 7) -> dict[str, list[str]]:
    """
    Cenario atual: cada time escolhe seus dias de forma independente, seguindo a
    distribuicao de mercado. Ninguem coordena com ninguem. E o equilibrio ruim.
    """
    rng = random.Random(seed)
    politica = {e.id: e for e in cenario.empresas}
    escala: dict[str, list[str]] = {}

    for t in cenario.times:
        k = politica[t.empresa].dias_min
        escolhidos: set[str] = set()
        if t.dia_ancora:
            escolhidos.add(t.dia_ancora)
        while len(escolhidos) < k:
            escolhidos.add(rng.choices(DIAS, weights=PESO_DIA_BASELINE, k=1)[0])
        escala[t.id] = sorted(escolhidos, key=DIAS.index)

    return escala
