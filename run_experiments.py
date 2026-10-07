import mlflow
import torch
import torch.nn as nn

from src.data import load_data, set_seed
from src.model import BeanMLP
from src.train import train_one_epoch, validate


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_experiment("dry-bean-classification")


# ============================================================
# CONFIGURAÇÕES DOS EXPERIMENTOS
# ============================================================

EXPERIMENTS = [
    {
        "name": "A_referencia",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
    },
    {
        "name": "B_taxa_menor",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    },
    {
        "name": "C_com_L2",
        "learning_rate": 0.01,
        "weight_decay": 1e-5,
    },
]


def run_experiment(config):
    """
    Executa uma configuração completa de treinamento.
    """

    set_seed(SEED)

    with mlflow.start_run(run_name=config["name"]):

        # ----------------------------------------------------
        # PARÂMETROS DA RUN
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "input_size": 16,
                "hidden_size_1": 64,
                "hidden_size_2": 32,
                "num_classes": 7,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "split": "64% train / 16% validation / 20% test",
            }
        )

        # ----------------------------------------------------
        # PREPARAÇÃO DOS DADOS
        # ----------------------------------------------------

        with mlflow.start_span(name="prepare_data"):
            train_loader, val_loader, _, _, _ = load_data(
                batch_size=BATCH_SIZE,
                seed=SEED,
            )

        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        model = BeanMLP(
            input_size=16,
            hidden_size1=64,
            hidden_size2=32,
            num_classes=7,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        best_val_loss = float("inf")
        best_state = None

        # ----------------------------------------------------
        # TREINAMENTO
        # ----------------------------------------------------

        with mlflow.start_span(name="train"):

            for epoch in range(EPOCHS):

                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    DEVICE,
                )

                # --------------------------------------------
                # VALIDAÇÃO
                # --------------------------------------------

                with mlflow.start_span(name="validate"):

                    val_loss, val_accuracy, val_f1 = validate(
                        model,
                        val_loader,
                        criterion,
                        DEVICE,
                    )

                # Métricas por época
                mlflow.log_metric(
                    "train_loss",
                    train_loss,
                    step=epoch,
                )

                mlflow.log_metric(
                    "val_loss",
                    val_loss,
                    step=epoch,
                )

                mlflow.log_metric(
                    "val_accuracy",
                    val_accuracy,
                    step=epoch,
                )

                mlflow.log_metric(
                    "val_f1",
                    val_f1,
                    step=epoch,
                )

                # Guardar melhor modelo segundo validação
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

                print(
                    f'{config["name"]} | '
                    f'Epoch {epoch + 1:02d}/{EPOCHS} | '
                    f'Train Loss: {train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f} | '
                    f'Val Acc: {val_accuracy:.4f} | '
                    f'Val F1: {val_f1:.4f}'
                )

        # Restaurar melhor estado encontrado na validação
        model.load_state_dict(best_state)

        # ----------------------------------------------------
        # VALIDAÇÃO FINAL DA RUN
        # ----------------------------------------------------

        with mlflow.start_span(name="final_validation"):

            final_val_loss, final_val_accuracy, final_val_f1 = validate(
                model,
                val_loader,
                criterion,
                DEVICE,
            )

        mlflow.log_metric("best_val_loss", final_val_loss)
        mlflow.log_metric("best_val_accuracy", final_val_accuracy)
        mlflow.log_metric("best_val_f1", final_val_f1)

        # Registrar modelo como artefato da run
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        print()
        print(f'Run "{config["name"]}" concluída.')
        print(f"Melhor Val Loss: {final_val_loss:.4f}")
        print(f"Val Accuracy: {final_val_accuracy:.4f}")
        print(f"Val F1: {final_val_f1:.4f}")
        print("-" * 70)


def main():

    print("=" * 70)
    print("DRY BEAN CLASSIFICATION - MLP + MLFLOW")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Experimentos: {len(EXPERIMENTS)}")
    print()

    for config in EXPERIMENTS:
        run_experiment(config)

    print()
    print("=" * 70)
    print("TODOS OS EXPERIMENTOS FORAM FINALIZADOS")
    print("=" * 70)


if __name__ == "__main__":
    main()