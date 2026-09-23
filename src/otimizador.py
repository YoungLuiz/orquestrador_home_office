"""
Otimizador de escala presencial via programacao por restricoes (CP-SAT).

FORMULACAO
----------
Variavel de decisao
    x[t, d] in {0, 1}   time t esta presencial no dia d

Grandeza derivada
    carga[r, d] = SOMA_t  n[t, r] * x[t, d]
    onde n[t, r] e quantas pessoas do time t chegam pelo recurso r

Funcao objetivo (minimizar)
    W_PICO * pico_permil
  + W_EXC  * SOMA_{r,d} excesso[r, d] * peso[r]
  + W_PREF * SOMA_t (dias preferidos nao atendidos)

    pico_permil  = maior razao carga/capacidade observada, em permilagem
    excesso[r,d] = max(0, carga[r,d] - capacidade[r])
    peso[r]      = 10000 / capacidade[r], normaliza recursos de portes diferentes

Restricoes
    R1  dias_min[e] <= SOMA_d x[t,d] <= dias_max[e]        politica da empresa
    R2  x[t, ancora] = 1                                   ritual fixo do time
    R3  SOMA_{t in e} hc[t] * x[t,d] <= postos[e]          capacidade do escritorio
    R4  existe pelo menos 1 dia em que todos os times       dia comum (opcional)
        da empresa e estao presentes

POR QUE CP-SAT E NAO REDE NEURAL
--------------------------------
Isto e um problema de otimizacao combinatoria com restricoes duras, nao um
problema de predicao. O CP-SAT devolve a solucao junto com o gap de otimalidade,
ou seja, voce consegue afirmar na banca o quao perto do otimo global chegou.
Machine learning entra em outra camada: aprender a curva carga -> congestionamento
que hoje esta simplificada como capacidade[r] fixa.
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .dados import DIAS, Cenario


@dataclass
class Pesos:
    pico: int = 100
    excesso: int = 1
    preferencia: int = 5
    posto: int = 40


@dataclass
class Resultado:
    escala: dict[str, list[str]]
    status: str
    viavel: bool
    pico_permil: int | None
    tempo_s: float
    gap_info: str
    sem_posto: int = 0


def diagnosticar(cenario: Cenario, fator_postos: float) -> list[str]:
    """
    Checagem estrutural antes de chamar o solver.

    Se uma empresa exige dias_min dias presenciais e so tem postos para uma
    fracao f do headcount por dia, ela precisa de dias_min pessoas-dia por
    pessoa e so dispoe de 5*f. Logo, o modelo so pode ser viavel se
        fator_postos >= dias_min / 5
    Vale a pena detectar isso aqui: o CP-SAT diria apenas INFEASIBLE, sem
    apontar a causa.
    """
    problemas: list[str] = []
    for e in cenario.empresas:
        minimo = e.dias_min / len(DIAS)
        if fator_postos < minimo - 1e-9:
            problemas.append(
                f"{e.nome}: exige {e.dias_min} dias presenciais, o que requer "
                f"pelo menos {minimo:.0%} dos postos (configurado: {fator_postos:.0%})"
            )
    return problemas


def otimizar(
    cenario: Cenario,
    pesos: Pesos | None = None,
    exigir_dia_comum: bool = True,
    fator_postos: float = 1.0,
    limite_s: float = 20.0,
    workers: int = 8,
) -> Resultado:
    pesos = pesos or Pesos()

    problemas = diagnosticar(cenario, fator_postos)
    if problemas:
        amostra = "; ".join(problemas[:3])
        extra = f" (+{len(problemas) - 3} empresas)" if len(problemas) > 3 else ""
        return Resultado(
            escala={},
            status="INVIAVEL_POR_CONSTRUCAO",
            viavel=False,
            pico_permil=None,
            tempo_s=0.0,
            gap_info=f"{amostra}{extra}",
        )

    m = cp_model.CpModel()

    times = cenario.times
    empresas = {e.id: e for e in cenario.empresas}
    recursos = cenario.recursos

    # ---------- variaveis ----------
    x = {
        (t.id, d): m.NewBoolVar(f"x_{t.id}_{d}")
        for t in times
        for d in DIAS
    }

    # ---------- R1: dias por semana ----------
    for t in times:
        e = empresas[t.empresa]
        total_dias = sum(x[t.id, d] for d in DIAS)
        m.Add(total_dias >= e.dias_min)
        m.Add(total_dias <= e.dias_max)

    # ---------- R2: dia ancora ----------
    for t in times:
        if t.dia_ancora:
            m.Add(x[t.id, t.dia_ancora] == 1)

    # ---------- R3: postos no escritorio (restricao FLEXIVEL) ----------
    # Times sao blocos indivisiveis de 18 a 60 pessoas, entao exigir que a soma
    # caiba exatamente sob o limite vira um bin packing que costuma nao ter
    # solucao. Na pratica o excedente vira hot desk, cafe ou sala de reuniao.
    # Modelamos como folga penalizada: o solver so estoura se nao houver
    # alternativa, e o app reporta quantas pessoas ficaram sem posto.
    faltas = []
    for e in cenario.empresas:
        limite = max(1, int(e.postos * fator_postos))
        for d in DIAS:
            folga = m.NewIntVar(0, e.postos, f"falta_{e.id}_{d}")
            m.Add(
                sum(t.headcount * x[t.id, d] for t in cenario.times_da(e.id))
                <= limite + folga
            )
            faltas.append(folga)

    # ---------- R4: dia comum por empresa ----------
    if exigir_dia_comum:
        for e in cenario.empresas:
            ts = cenario.times_da(e.id)
            comum = []
            for d in DIAS:
                y = m.NewBoolVar(f"comum_{e.id}_{d}")
                for t in ts:
                    m.AddImplication(y, x[t.id, d])
                comum.append(y)
            m.Add(sum(comum) >= 1)

    # ---------- carga por recurso e dia ----------
    carga: dict[tuple[str, str], cp_model.IntVar] = {}
    for r, cap in recursos.items():
        teto = sum(t.por_recurso.get(r, 0) for t in times)
        for d in DIAS:
            v = m.NewIntVar(0, max(teto, 1), f"carga_{r}_{d}")
            m.Add(
                v == sum(
                    t.por_recurso.get(r, 0) * x[t.id, d]
                    for t in times
                    if t.por_recurso.get(r, 0)
                )
            )
            carga[r, d] = v

    # ---------- termo de pico (minimax normalizado) ----------
    # pico_permil * cap[r] >= 1000 * carga[r,d]  para todo (r, d).
    # Como cap[r] e constante, a restricao permanece linear.
    pico = m.NewIntVar(0, 5000, "pico_permil")
    for (r, d), v in carga.items():
        m.Add(pico * recursos[r] >= 1000 * v)

    # ---------- termo de excesso acima da capacidade ----------
    termos_excesso = []
    for (r, d), v in carga.items():
        cap = recursos[r]
        exc = m.NewIntVar(0, 10**6, f"exc_{r}_{d}")
        m.Add(exc >= v - cap)
        peso_r = max(1, round(10000 / cap))
        termos_excesso.append(exc * peso_r)

    # ---------- termo de preferencia ----------
    termos_pref = []
    for t in times:
        for d in t.dias_preferidos:
            termos_pref.append(1 - x[t.id, d])

    m.Minimize(
        pesos.pico * pico
        + pesos.excesso * sum(termos_excesso)
        + pesos.preferencia * sum(termos_pref)
        + pesos.posto * sum(faltas)
    )

    # ---------- resolucao ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limite_s
    solver.parameters.num_search_workers = workers
    status = solver.Solve(m)

    nome_status = solver.StatusName(status)
    viavel = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    if not viavel:
        return Resultado(
            escala={},
            status=nome_status,
            viavel=False,
            pico_permil=None,
            tempo_s=solver.WallTime(),
            gap_info="sem solucao: relaxe postos, dias_min ou o dia comum",
        )

    escala = {
        t.id: [d for d in DIAS if solver.Value(x[t.id, d])]
        for t in times
    }

    if status == cp_model.OPTIMAL:
        gap = "otimo global provado"
    else:
        gap = (
            f"melhor solucao ate {limite_s:.0f}s | "
            f"limite inferior {solver.BestObjectiveBound():.0f} | "
            f"objetivo {solver.ObjectiveValue():.0f}"
        )

    return Resultado(
        escala=escala,
        sem_posto=sum(solver.Value(f) for f in faltas),
        status=nome_status,
        viavel=True,
        pico_permil=solver.Value(pico),
        tempo_s=solver.WallTime(),
        gap_info=gap,
    )
