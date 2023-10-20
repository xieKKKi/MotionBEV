# -*- coding:utf-8 -*-
# author: Xinge

from pathlib import Path

from strictyaml import Bool, Float, Int, Map, Seq, Str, as_document, load

model_params = Map(
    {
        "grid_size": Seq(Int()),
        "fea_dim": Int(),
        "ppmodel_init_dim": Int(),
        "use_norm": Bool(),
        "dropout": Float(),
        "use_co_attention": Bool(),
    }
)


data_loader = Map(
    {
        "data_path": Str(),
        "return_ref": Bool(),
        "residual": Int(),
        "residual_path": Str(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "drop_few_static_frames": Bool(),
        "num_workers": Int(),
        "dataset_type": Str(),
        "ignore_label": Int(),
        "fixed_volume_space": Bool(),
        "rotate_aug": Bool(),
        "flip_aug": Bool(),
        "transform_aug": Bool(),
    }
)

train_params = Map(
    {
        "name": Str(),
        "model_load_path": Str(),
        "checkpoint_every_n_steps": Int(),
        "max_num_epochs": Int(),
        "eval_every_n_steps": Int(),
        "optimizer": Str(),
        "learning_rate": Float(),
        "weight_decay": Float(),
        "momentum": Float(),
        "wup_epochs": Float(),
        "lr_decay": Float(),
    }
)

schema_v4 = Map(
    {
        "format_version": Int(),
        "model_params": model_params,
        "data_loader": data_loader,
        "train_params": train_params,
    }
)

SCHEMA_FORMAT_VERSION_TO_SCHEMA = {4: schema_v4}


def load_config_data(path: str) -> dict:
    yaml_string = Path(path).read_text()
    cfg_without_schema = load(yaml_string, schema=None)
    schema_version = int(cfg_without_schema["format_version"])
    if schema_version not in SCHEMA_FORMAT_VERSION_TO_SCHEMA:
        raise Exception(f"Unsupported schema format version: {schema_version}.")

    strict_cfg = load(yaml_string, schema=SCHEMA_FORMAT_VERSION_TO_SCHEMA[schema_version])
    cfg: dict = strict_cfg.data
    return cfg


def config_data_to_config(data):  # type: ignore
    return as_document(data, schema_v4)


def save_config_data(data: dict, path: str) -> None:
    cfg_document = config_data_to_config(data)
    with open(Path(path), "w") as f:
        f.write(cfg_document.as_yaml())
