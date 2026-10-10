import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import plotly.express as px

    from network.model import DEFAULT_MODEL_PATH, load_model
    from network.training import EpochMetrics
    from pipeline.dataset import DigitDataset
    from pipeline.figures import confusion_matrix_figure, training_history_figure
    from pipeline.printed import generate_printed_digits
    from pipeline.sources import load_mnist
    from pipeline.transforms import confusion_matrix, digit_image

    return (
        DEFAULT_MODEL_PATH, DigitDataset, EpochMetrics, confusion_matrix, confusion_matrix_figure,
        digit_image, generate_printed_digits, json, load_mnist, load_model, mo, px,
        training_history_figure,
    )


@app.cell
def _(mo):
    mo.md("""
    # Vision : analyse du modèle

    Le modèle est entraîné par `uv run python -m network.train` (backend/) ; ce notebook lit le
    modèle enregistré et son historique, puis le mesure sur le jeu de test (MNIST et chiffres
    imprimés, jamais vus pendant l'entraînement).
    """)
    return


@app.cell
def _(DEFAULT_MODEL_PATH, EpochMetrics, json, training_history_figure):
    history_path = DEFAULT_MODEL_PATH.with_name("historique.json")
    history = [EpochMetrics(**metrics) for metrics in json.loads(history_path.read_text())]
    training_history_figure(history)
    return


@app.cell
def _(DEFAULT_MODEL_PATH, DigitDataset, generate_printed_digits, load_mnist, load_model):
    model = load_model(DEFAULT_MODEL_PATH)
    test = DigitDataset.concatenate(load_mnist("test"), generate_printed_digits(100, seed=1))
    predictions = model.predict(test.images)
    return predictions, test


@app.cell
def _(confusion_matrix, confusion_matrix_figure, mo, predictions, test):
    test_accuracy = (predictions == test.labels).mean()
    mo.vstack([
        mo.md(f"**Exactitude sur le jeu de test : {test_accuracy:.2%}** ({len(test)} images)"),
        confusion_matrix_figure(confusion_matrix(test.labels, predictions)),
    ])
    return


@app.cell
def _(digit_image, mo, predictions, px, test):
    errors = (predictions != test.labels).nonzero()[0][:12]
    mo.hstack([
        mo.vstack([
            px.imshow(digit_image(test.images[index]), color_continuous_scale="gray", width=160, height=160)
            .update_layout(coloraxis_showscale=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}),
            mo.md(f"réel {test.labels[index]}, prédit {predictions[index]}"),
        ])
        for index in errors
    ], wrap=True)
    return


if __name__ == "__main__":
    app.run()
