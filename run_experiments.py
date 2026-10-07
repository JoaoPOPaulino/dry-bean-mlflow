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

        # ====================================================
        # TRACE PRINCIPAL DO PIPELINE
        # ====================================================

        with mlflow.start_span(name="pipeline") as pipeline_span:

            pipeline_span.set_attributes(
                {
                    "run_name": config["name"],
                    "learning_rate": config["learning_rate"],
                    "weight_decay": config["weight_decay"],
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                    "device": str(DEVICE),
                }
            )

            # ------------------------------------------------
            # PARÂMETROS DA RUN
            # ------------------------------------------------

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

            # ------------------------------------------------
            # PREPARAÇÃO DOS DADOS
            # ------------------------------------------------

            with mlflow.start_span(name="prepare_data") as prepare_span:

                train_loader, val_loader, _, _, _ = load_data(
                    batch_size=BATCH_SIZE,
                    seed=SEED,
                )

                prepare_span.set_attributes(
                    {
                        "train_samples": len(train_loader.dataset),
                        "validation_samples": len(val_loader.dataset),
                        "batch_size": BATCH_SIZE,
                    }
                )

            # ------------------------------------------------
            # MODELO
            # ------------------------------------------------

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
            best_epoch = None

            # ------------------------------------------------
            # TREINAMENTO
            # ------------------------------------------------

            with mlflow.start_span(name="train") as train_span:

                for epoch in range(EPOCHS):

                    train_loss = train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        DEVICE,
                    )

                    # ----------------------------------------
                    # VALIDAÇÃO DA ÉPOCA
                    # ----------------------------------------

                    with mlflow.start_span(
                        name=f"validate_epoch_{epoch + 1:02d}"
                    ) as validation_span:

                        val_loss, val_accuracy, val_f1 = validate(
                            model,
                            val_loader,
                            criterion,
                            DEVICE,
                        )

                        validation_span.set_attributes(
                            {
                                "epoch": epoch + 1,
                                "val_loss": float(val_loss),
                                "val_accuracy": float(val_accuracy),
                                "val_f1": float(val_f1),
                            }
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

                    # Melhor modelo segundo a perda de validação
                    if val_loss < best_val_loss:

                        best_val_loss = val_loss
                        best_epoch = epoch + 1

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

                train_span.set_attributes(
                    {
                        "best_epoch": best_epoch,
                        "best_val_loss": float(best_val_loss),
                    }
                )

            # ------------------------------------------------
            # RESTAURAR MELHOR MODELO
            # ------------------------------------------------

            model.load_state_dict(best_state)

            # ------------------------------------------------
            # VALIDAÇÃO FINAL
            # ------------------------------------------------

            with mlflow.start_span(
                name="final_validation"
            ) as final_validation_span:

                final_val_loss, final_val_accuracy, final_val_f1 = validate(
                    model,
                    val_loader,
                    criterion,
                    DEVICE,
                )

                final_validation_span.set_attributes(
                    {
                        "best_epoch": best_epoch,
                        "val_loss": float(final_val_loss),
                        "val_accuracy": float(final_val_accuracy),
                        "val_f1": float(final_val_f1),
                    }
                )

            # ------------------------------------------------
            # MÉTRICAS FINAIS
            # ------------------------------------------------

            mlflow.log_metric(
                "best_val_loss",
                final_val_loss,
            )

            mlflow.log_metric(
                "best_val_accuracy",
                final_val_accuracy,
            )

            mlflow.log_metric(
                "best_val_f1",
                final_val_f1,
            )

            mlflow.log_metric(
                "best_epoch",
                best_epoch,
            )

            # ------------------------------------------------
            # MODELO
            # ------------------------------------------------

            mlflow.pytorch.log_model(
                model,
                name="model",
                serialization_format="pickle",
            )

            # Informações finais do trace
            pipeline_span.set_attributes(
                {
                    "selected_epoch": best_epoch,
                    "final_val_loss": float(final_val_loss),
                    "final_val_accuracy": float(final_val_accuracy),
                    "final_val_f1": float(final_val_f1),
                }
            )

        # ====================================================
        # RESULTADO DA RUN
        # ====================================================

        print()
        print(f'Run "{config["name"]}" concluída.')
        print(f"Melhor época: {best_epoch}")
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