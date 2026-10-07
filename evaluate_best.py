import torch
import torch.nn as nn
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data import load_data
from src.model import BeanMLP


SEED = 42
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    _, _, test_loader, label_encoder, _ = load_data(
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # Localizar a Run B no MLflow
    experiment = mlflow.get_experiment_by_name(
        "dry-bean-classification"
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

    # Carregar exatamente o modelo salvo pela Run B
    model_uri = f"runs:/{run_id}/model"

    model = mlflow.pytorch.load_model(
        model_uri,
        map_location=DEVICE,
    )

    model = model.to(DEVICE)

    # Avaliação única no conjunto de teste
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

    print()
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Macro: {f1:.4f}")

    print()
    print("MATRIZ DE CONFUSÃO")
    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    print()
    print("RELATÓRIO DE CLASSIFICAÇÃO")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()