# /// script
# dependencies = [
#     "brahe==1.1.4",
#     "marimo",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "polars==1.39.3",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    return np, pl


@app.cell
def _(pl):
    data = pl.read_json("data.json")
    data = data.select(pl.col("INCLINATION").alias("inc"), 
                pl.col("RA_OF_ASC_NODE").alias("raan"), 
                pl.col("MEAN_ANOMALY").alias("phase"),
    )
    return (data,)


@app.cell
def _(data, np, pl):
    test = np.concat([np.random.rand(1) * 180, np.random.rand(2, ) * 360])
    print(test)
    data.filter((pl.col("inc") < 90) == (test[0] < 90)).select(
        [np.abs(pl.col(name) - val) for name, val in zip(data.columns, test)]
    ).with_columns(mass=200).with_columns(
        cost=pl.col("mass") - (pl.col("inc") + pl.col("raan")) * 2 - pl.col("phase")
    ).sort(by="cost", descending=True)
    return


if __name__ == "__main__":
    app.run()
