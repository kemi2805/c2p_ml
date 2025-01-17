from src.utils.arg_parser import parse_args_and_config
from src.models.nnc2p import NNC2P
from src.data.data_generator import HybridPiecewiseDataGenerator, TabulatedDataGenerator
from src.training.trainer import Trainer
from src.utils.helpers import get_device, ensure_directories

def get_model(config):
    model_name = config["model"]["name"]
    # Check if the model name is valid
    valid_model_names = ["NNC2PS", "NNC2PL", "NNC2P_Tabulated"]
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
    else:
        raise ValueError(f"Unknown model type for data generation: {model_name}")

def main():
    args, config = parse_args_and_config()
    print(config)
    from sys import exit
    exit()

    # Ensure required directories exist
    ensure_directories(config)

    device = get_device()
    model = get_model(config)
    data_generator = get_data_generator(config)
    train_loader, val_loader, test_loader = data_generator.get_data_loaders()

    trainer = Trainer(model, config, device)

    # Train model
    trainer.train(train_loader, val_loader, data_generator.output_scaler)

if __name__ == "__main__":
    main()
