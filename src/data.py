import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(
    file_path="data/Dry_Bean_Dataset.xlsx",
    batch_size=64,
    seed=42,
):
    set_seed(seed)

    # 1. Carregar dataset
    df = pd.read_excel(file_path)

    # 2. Separar atributos e classe
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # 3. Converter as classes de texto para números
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # 4. Separar 20% para teste final
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    # 5. Dos 80% restantes, separar 20% para validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=seed,
        stratify=y_train_val,
    )

    # 6. Ajustar scaler SOMENTE com os dados de treinamento
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Validação e teste usam o scaler aprendido no treino
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 7. Converter para tensores PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 8. Criar datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # 9. Criar DataLoaders
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        label_encoder,
        scaler,
    )
