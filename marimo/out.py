# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.21.1",
#     "matplotlib==3.10.8",
#     "networkx==3.6.1",
#     "numpy==2.4.3",
#     "polars==1.39.3",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import polars as pl
    import time 
    from itertools import combinations
    import matplotlib.pyplot as plt
    import heapq

    return heapq, np, pl, plt, time


@app.cell
def _(np):
    def d_ang(a, b):
        return abs(((a - b + 180) % 360) - 180)

    def build_cost_matrix(inc, raan, phase, m):
        n = len(inc)
        C = np.empty((n, n), dtype=float)

        for k in range(n):
            for j in range(n):
                C[k, j] = -((
                m[j]
                - 2 * (d_ang(inc[j], inc[k]) + d_ang(raan[j], raan[k]))
                - d_ang(phase[j], phase[k])


                ))
        np.fill_diagonal(C,np.inf)
        return C

    return (d_ang,)


@app.cell
def _(d_ang, np):
    def get_cost(inc, raan, phase, mass):
        inc = np.deg2rad(inc)
        raan = np.deg2rad(raan)
        phase = np.deg2rad(phase)

        return 2 * np.sin(0.5 * np.sqrt(np.square(inc) + np.square(raan)) 
                   + 1 - np.cos(phase)
                   - 2 * np.log(mass))
    def build_cost_matrix_deluxe(inc, raan, phase, m):
        n = len(inc)
        C = np.empty((n, n), dtype=float)

        for k in range(n):
            for j in range(n):
                C[k,j] = get_cost(
                    d_ang(inc[j], inc[k]), 
                    d_ang(raan[j], raan[k]), 
                    d_ang(phase[j], phase[k]), 
                    m[j]
                )
        np.fill_diagonal(C,np.inf)
        return C

    return (build_cost_matrix_deluxe,)


@app.cell
def _(np):
    def random_exponential_mass(lam, n, m_min, m_max):
        """
        Genera n muestras de una distribución exponencial truncada decreciente
        en el intervalo [m_min, m_max].

        Parámetros
        ----------
        lam : float
            Tasa de decaimiento (> 0).
        n : int
            Número de muestras.
        m_min : float
            Valor mínimo del intervalo.
        m_max : float
            Valor máximo del intervalo.

        Retorna
        -------
        np.ndarray
            Array de tamaño n con las muestras generadas.
        """
        if lam <= 0:
            raise ValueError("lam debe ser > 0")
        if n <= 0:
            raise ValueError("n debe ser > 0")
        if m_min >= m_max:
            raise ValueError("m_min debe ser menor que m_max")

        u = np.random.rand(n)
        x = m_min - (1 / lam) * np.log(
            1 - u * (1 - np.exp(-lam * (m_max - m_min)))
        )
        return x

    return (random_exponential_mass,)


@app.cell
def _(pl):
    data = pl.read_json("data.json").select(
            pl.col("INCLINATION").alias("inc"),
            pl.col("RA_OF_ASC_NODE").alias("raan"),
            pl.col("MEAN_ANOMALY").alias("phase"),
    ).with_row_index("id")

    data
    # np.shape(data)
    return (data,)


@app.cell
def _(data, np, pl):
    test = np.concat([np.random.rand(1) * 180, np.random.rand(2, ) * 360])

    result = (
        data
        .filter((pl.col("inc") < 90) == (test[0] < 90)) #Elimina elementos que rotan en direccion contraria a nuestro basurero
        .select([pl.col("id"),
            (pl.col("inc") - test[0]).abs().alias("inc"),
            (((pl.col("raan") - test[1] + 180) % 360) - 180).abs().alias("raan"),
            (((pl.col("phase") - test[2] + 180) % 360) - 180).abs().alias("phase")

        ])
        .with_columns(
            (
                pl.lit(200)
                - pl.col("inc")
                - pl.col("phase")
            ).alias("cost")
        )
        .sort("cost", descending=True)
    )

    result
    return result, test


@app.cell
def _(data, np, pl, random_exponential_mass, result, test):
    # ti = time.time()
    result_1 = (
        data
        .filter((pl.col('inc') < 90) == (test[0] < 90))
        .select([
            pl.col('id'),
            (pl.col('inc') - test[0]).abs().alias('inc'),
            ((pl.col('raan') - test[1] + 180) % 360 - 180).abs().alias('raan'),
            ((pl.col('phase') - test[2] + 180) % 360 - 180).abs().alias('phase')
        ])
        .with_columns(
            (pl.lit(1 / 0.7) - 2 * np.sin(0.5 * (np.deg2rad(pl.col('inc')) ** 2 + np.deg2rad(pl.col('raan')) ** 2) ** 0.5) - (1 - np.cos(np.deg2rad(pl.col('phase'))))).alias('cost')
        )
        .sort('cost', descending=True)
        .with_columns(mass = random_exponential_mass(0.7, len(result), 0.193, 2 * np.pi) * 400 / 0.193)
    )

    result_1  #Elimina elementos que rotan en direccion contraria a nuestro basurero  #La masa está fija. Queremos incorporar la distribución exponencial que está abajo. La idea es a cada elemento de la lista inicial darle un valor de masa asociado al objeto (ID)
    result_1.select(pl.all().round(decimals=3))
    return (result_1,)


@app.cell
def _(data, pl, result_1):
    N = 1000
    result_100 = result_1.head(N)
    ids_100_df = result_100.select('id')
    data_100 = ids_100_df.join(data, on='id', how='left')
    data_100 = data_100.with_columns(pl.col('id').cast(pl.Int64))
    # Coordenadas de la casilla de salida: ESCALARES
    data_100
    return N, data_100


@app.cell
def _(N, np, random_exponential_mass):
    #Mass distribution (Exponential)
    # Parámetros
    m_min_norm = 0.193 #Masa minima equivalente a 400kg
    m_max_norm = 2 * np.pi #Masa maxima equivalente a 13000 kg
    lam = 0.7      # tasa de decaimiento; mayor => más concentración cerca de a
    #N se ha definido en una celda mas arriba como el numero de muestras que nos quedamos
    m = random_exponential_mass(lam, N, m_min_norm, m_max_norm)
    m[:10]
    return (m,)


@app.cell
def _(data_100):
    data_100.with_columns()
    return


@app.cell
def _(build_cost_matrix_deluxe, data_100, m):
    #Cost matrix (Deluxe :P)
    inc = data_100['inc'].to_numpy()
    raan = data_100['raan'].to_numpy()
    phase = data_100['phase'].to_numpy()
    costdel = build_cost_matrix_deluxe(inc, raan, phase, m)
    return (costdel,)


@app.cell
def _(heapq, np):
    # start: Index of starting orbit
    # costs: full cost matrix (the full f-cost)
    # values: the "benefits" only (in this case the mass)
    # n_steps: how many steps to take
    # K: number of closest neighbors to look at
    def astar(start, costs, values, n_steps, K):
        N = len(values)
        nodes = np.arange(N)

        # Heuristic (prefix sum of best values)
        optimistic = np.sort(values)[::-1]
        optimistic = np.concatenate(([0], np.cumsum(optimistic)))

        heap = []
        heapq.heappush(heap, (
            -(values[start] + optimistic[n_steps]),
            start,
            0,
            values[start],
            frozenset([start])
        ))

        best_score = -np.inf
        best_path = None

        while heap:
            f, node, steps, g, visited = heapq.heappop(heap)

            if steps == n_steps:
                if g > best_score:
                    best_score = g
                    best_path = visited
                continue

            steps_left = n_steps - steps

            # Prune
            if g + optimistic[steps_left] <= best_score:
                continue

            # Candidates (slow but correct)
            candidates = np.array([i for i in nodes if i not in visited])
            if len(candidates) == 0:
                continue

            # Use VALUES (not costs) for scoring
            scores = costs[node, candidates]

            # Top-K
            idx = np.argpartition(scores, -K)[-K:]
            idx = idx[np.argsort(scores[idx])[::-1]]

            for i in idx:
                nb = candidates[i]

                new_g = g + costs[node, nb]
                new_visited = visited | {nb}

                h = optimistic[steps_left - 1]
                f = new_g + h

                heapq.heappush(
                    heap,
                    (-f, nb, steps + 1, new_g, new_visited)
                )

        return best_score, best_path

    return (astar,)


@app.cell
def _(astar, costdel, m, time):
    # k_steps = 
    start = 0
    tvector = []
    for i in [1, 2, 3, 4, 5, 6, 7]:
        _ti = time.time()
        best_cost, path = astar(0, costdel, m, i, 8)
        _tf = time.time()
        tvector.append(_tf - _ti)
        print(f'Coste = {best_cost},Camino seguido = {path},Tiempo tardado en computar = {_tf - _ti} s')
        #print(f'Coste_malo = {best_cost2},Camino seguido_malo = {path2},Tiempo tardado en computar_malo = {_tf - _ti} s')
    return (tvector,)


@app.cell
def _(tvector):
    print(tvector)
    return


@app.cell
def _(np, plt, tvector):
    # datos
    y = np.array([0.0024788379669189453, 0.04912734031677246, 13.047961473464966, 227.30829071998596])
    x = np.array([1, 2, 3, 4])

    # ajuste exponencial: y = a * exp(bx)
    b, log_a = np.polyfit(x, np.log(y), 1)
    a = np.exp(log_a)

    # curva ajustada
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = a * np.exp(b * x_fit)

    # plot
    plt.scatter(x, y, label="Fuerza bruta")
    #plt.plot(x_fit, y_fit, label="ajuste")

    plt.plot([1, 2, 3, 4, 5, 6, 7], tvector, 'b.', label = "A*")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    app.run()
