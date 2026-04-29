import gdown, os, torch
from pathlib import Path
from ..label_mapper import id2label_dict

model_urls = {
    "UNet_ResNet50_default": "https://drive.google.com/file/d/1Y9zubvMzkYHoAqz-NvV6vniH5FKAF2iV/view?usp=drive_link"
}

local_weight_files = {
    "UNet_ResNet50_default": "UNet_resnet50_default.pth",
}


def _get_store_path():
    if "CXAS_PATH" in os.environ:
        return os.path.join(os.environ["CXAS_PATH"], ".cxas")
    return os.path.join(os.path.expanduser("~"), ".cxas")


def _get_repo_weight_path(model_name: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    weight_filename = local_weight_files.get(model_name, model_name + ".pth")
    return str(repo_root / "weights" / weight_filename)


def _get_cached_weight_path(model_name: str) -> str:
    store_path = _get_store_path()
    return os.path.join(store_path, "weights", model_name + ".pth")


def _resolve_weight_path(model_name: str) -> str:
    if "CXAS_MODEL_PATH" in os.environ:
        env_path = os.environ["CXAS_MODEL_PATH"]
        if os.path.isfile(env_path):
            return env_path

    repo_weight_path = _get_repo_weight_path(model_name)
    if os.path.isfile(repo_weight_path):
        return repo_weight_path

    cached_weight_path = _get_cached_weight_path(model_name)
    if os.path.isfile(cached_weight_path):
        return cached_weight_path

    return cached_weight_path


def get_model(model_name, gpus=""):
    """
    Function to load a model by name and optionally move it to specified GPUs.

    Args:
        model_name (str): Name of the model to load.
        gpus (str): String containing GPU device IDs separated by commas.
                    If empty or 'cpu', the model will be loaded on CPU.

    Returns:
        torch.nn.Module: Loaded model.
    """
    assert model_name.split("_")[0] in list(model_getter.keys())

    model = model_getter[model_name.split("_")[0]](model_name)
    download_weights(model_name)
    model = load_weights(
        model, model_name, map_location="cpu" if "cpu" in gpus else "cuda"
    )

    if "cpu" not in gpus:
        gpus = [int(i) for i in gpus.split(",") if len(i) > 0]

        if len(gpus) > 1:
            assert torch.cuda.is_available()
            model.to(gpus[0])
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        elif len(gpus) == 1:
            assert torch.cuda.is_available()
            model.to(gpus[0])

    return model


def get_unet(model_name):
    """
    Function to get a U-Net model.

    Args:
        model_name (str): Name of the U-Net model.

    Returns:
        torch.nn.Module: U-Net model.
    """
    from .UNet.backbone_unet import BackboneUNet

    return BackboneUNet(model_name, len(id2label_dict.keys()))


def download_weights(model_name: str) -> None:
    """
    Function to download model weights.

    Args:
        model_name (str): Name of the model.
    """
    repo_weight_path = _get_repo_weight_path(model_name)
    if os.path.isfile(repo_weight_path):
        return

    store_path = _get_store_path()
    os.makedirs(os.path.join(store_path, "weights"), exist_ok=True)
    out_path = _get_cached_weight_path(model_name)
    if os.path.isfile(out_path):
        return
    else:
        gdown.download(model_urls[model_name], out_path, quiet=False, fuzzy=True)
        return


def load_weights(model, model_name: str, map_location: str = "cuda:0"):
    """
    Function to load model weights.

    Args:
        model (torch.nn.Module): Model to load weights into.
        model_name (str): Name of the model.
        map_location (str): Location to map tensors to (default: 'cuda:0' if available, else 'cpu').

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    out_path = _resolve_weight_path(model_name)
    assert os.path.isfile(out_path), f"Weight file not found for {model_name}: {out_path}"

    checkpoint = torch.load(out_path, map_location=map_location, weights_only=False)

    if "module" in list(checkpoint["model"].keys())[0]:
        for i in list(checkpoint["model"].keys()):
            checkpoint["model"][i[len("module.") :]] = checkpoint["model"].pop(i)
    model.load_state_dict(checkpoint["model"], strict=False)
    return model


# Dictionary containing model getter functions
model_getter = {
    "UNet": get_unet,
}
