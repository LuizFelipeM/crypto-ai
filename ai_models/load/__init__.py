from keras.models import Model, load_model, model_from_json


def load_complete_model(folder_path: str) -> Model:
    loaded_model: Model = load_model(f"{folder_path}/model.h5")  # type: ignore
    return loaded_model


def load_architecture(folder_path: str, load_weights=False) -> Model:
    with open(f"{folder_path}/model_architecture.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model: Model = model_from_json(loaded_model_json)  # type: ignore

    if load_weights:
        loaded_model.load_weights(f"{folder_path}/model_weights.h5")

    return loaded_model
