"""
Avaliacao de uma escala: transforma decisoes em carga por recurso/dia e
calcula as metricas que voce vai defender na banca.

Metricas
    pico_pct        maior razao carga/capacidade em todo o cluster
    celulas_acima   quantos pares (recurso, dia) passam de 100% da capacidade
    pessoas_exc     soma de pessoas alem da capacidade, o "excedente humano"
    desvio_dias     desvio padrao da carga total entre os 5 dias
                    (mede o quanto a semana ficou plana)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .dados import DIAS, Cenario


@dataclass
class Metricas:
    pico_pct: float
    recurso_pico: str
    dia_pico: str
    celulas_acima: int
    pessoas_exc: int
    desvio_dias: float
    carga_por_dia: dict[str, int]


def matriz_carga(cenario: Cenario, escala: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Retorna carga[recurso][dia] em pessoas."""
    mat = {r: {d: 0 for d in DIAS} for r in cenario.recursos}
    for t in cenario.times:
        for d in escala.get(t.id, []):
            for r, n in t.por_recurso.items():
                mat[r][d] += n
    return mat


def avaliar(cenario: Cenario, escala: dict[str, list[str]]) -> Metricas:
    mat = matriz_carga(cenario, escala)

    pico_pct, recurso_pico, dia_pico = 0.0, "", ""
    celulas_acima = 0
    pessoas_exc = 0

    for r, cap in cenario.recursos.items():
        for d in DIAS:
            carga = mat[r][d]
            pct = 100.0 * carga / cap
            if pct > pico_pct:
                pico_pct, recurso_pico, dia_pico = pct, r, d
            if carga > cap:
                celulas_acima += 1
                pessoas_exc += carga - cap

    carga_por_dia = {d: sum(mat[r][d] for r in cenario.recursos) for d in DIAS}
    desvio = statistics.pstdev(list(carga_por_dia.values()))

    return Metricas(
        pico_pct=pico_pct,
        recurso_pico=recurso_pico,
        dia_pico=dia_pico,
        celulas_acima=celulas_acima,
        pessoas_exc=pessoas_exc,
        desvio_dias=desvio,
        carga_por_dia=carga_por_dia,
    )


def comparar(antes: Metricas, depois: Metricas) -> dict[str, float]:
    def var(a: float, b: float) -> float:
        return 0.0 if a == 0 else 100.0 * (b - a) / a

    return {
        "pico_pct": var(antes.pico_pct, depois.pico_pct),
        "pessoas_exc": var(antes.pessoas_exc, depois.pessoas_exc),
        "desvio_dias": var(antes.desvio_dias, depois.desvio_dias),
    }
