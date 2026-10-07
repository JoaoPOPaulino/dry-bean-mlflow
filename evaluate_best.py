from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data import load_data


SEED = 42
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def evaluate(model, loader):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(y_batch.numpy())

    return targets, predictions


def main():
    print("=" * 60)
    print("AVALIAÇÃO FINAL - RUN B")
    print("=" * 60)

    # --------------------------------------------------------
    # Carregar conjunto de teste reservado
    # --------------------------------------------------------

    _, _, test_loader, label_encoder, _ = load_data(
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Localizar a Run B no MLflow
    # --------------------------------------------------------

    experiment = mlflow.get_experiment_by_name(
        "dry-bean-classification"
    )

    if experiment is None:
        raise RuntimeError(
            "Experimento 'dry-bean-classification' não encontrado."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'B_taxa_menor'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(
            "Não foi encontrada uma execução B_taxa_menor."
        )

    run_id = runs.iloc[0]["run_id"]

    print(f"Run selecionada: {run_id}")

    # --------------------------------------------------------
    # Carregar modelo escolhido
    # --------------------------------------------------------

    model_uri = f"runs:/{run_id}/model"

    model = mlflow.pytorch.load_model(
        model_uri,
        map_location=DEVICE,
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Avaliação no teste reservado
    # --------------------------------------------------------

    y_true, y_pred = evaluate(
        model,
        test_loader,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
    )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print()
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Macro: {f1:.4f}")

    print()
    print("MATRIZ DE CONFUSÃO")
    print(matrix)

    print()
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print(report)

    # --------------------------------------------------------
    # Salvar resultados
    # --------------------------------------------------------

    results_path = ARTIFACTS_DIR / "test_results.txt"

    with open(results_path, "w", encoding="utf-8") as file:
        file.write("AVALIAÇÃO FINAL - DRY BEAN CLASSIFICATION\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Run selecionada: {run_id}\n")
        file.write("Experimento selecionado: B_taxa_menor\n")
        file.write(f"Test Accuracy: {accuracy:.4f}\n")
        file.write(f"Test F1 Macro: {f1:.4f}\n")

    report_path = ARTIFACTS_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # --------------------------------------------------------
    # Matriz de confusão
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=label_encoder.classes_,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(
        "Matriz de Confusão - Run B - Conjunto de Teste"
    )

    fig.tight_layout()

    confusion_matrix_path = (
        ARTIFACTS_DIR / "confusion_matrix.png"
    )

    fig.savefig(
        confusion_matrix_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 60)
    print("ARTEFATOS GERADOS")
    print("=" * 60)
    print(results_path)
    print(report_path)
    print(confusion_matrix_path)


if __name__ == "__main__":
    main()