from src.utils.arg_parser import parse_args_and_config
from src.models.nnc2p import NNC2P
from src.data.data_generator import HybridPiecewiseDataGenerator, TabulatedDataGenerator, HybridPiecewiseDataGenerator_3D
from src.training.trainer import Trainer
from src.utils.helpers import get_device, ensure_directories

def get_model(config):
    model_name = config["model"]["name"]
    # Check if the model name is valid
    valid_model_names = ["NNC2PS", "NNC2PL", "NNC2P_Tabulated", "NNC2PS_3D"]
    if model_name not in valid_model_names:
        raise ValueError(f"Unknown model: {model_name}")
    # Return the single available model
    return NNC2P(config)

def get_data_generator(config):
    model_name = config["model"]["name"]
    if model_name in ["NNC2PS", "NNC2PL"]:
        return HybridPiecewiseDataGenerator(config)
    elif model_name == "NNC2P_Tabulated":
        return TabulatedDataGenerator(config)
    elif model_name in ["NNC2PS_3D"]:
        return HybridPiecewiseDataGenerator_3D(config)
    else:
        raise ValueError(f"Unknown model type for data generation: {model_name}")

def main():
    args, config = parse_args_and_config()

    # Ensure required directories exist
    ensure_directories(config)

    device = get_device()
    model = get_model(config)
    data_generator = get_data_generator(config)
    train_loader, val_loader, test_loader = data_generator.get_data_loaders()
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Train values: {train_loader.dataset[0]}")
    print("Transform Dataset")
    train_loader_numpy = train_loader.numpy()
    train_loader_transformed = train_loader_numpy.inverse_transform(train_loader_numpy)
    from sys import exit
    exit()

    trainer = Trainer(model, config, device)

    # Train model
    trainer.train(train_loader, val_loader, data_generator.output_scaler)

    trainer.plot_training_history(config["model"]["name"])

if __name__ == "__main__":
    main()
