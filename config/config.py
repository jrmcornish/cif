import json

from .dsl import CONFIG_GROUPS, CURRENT_CONFIG_GROUP, GridParams

from . import two_d, uci, images


def get_config_group(dataset):
    for group, group_data in CONFIG_GROUPS.items():
        if dataset in group_data["datasets"]:
            return group

    assert False, f"Dataset `{dataset}' not found"


def get_datasets():
    result = []
    for items in CONFIG_GROUPS.values():
        result += items["datasets"]
    return result


def get_models():
    result = []
    for items in CONFIG_GROUPS.values():
        result += list(items["model_configs"])
    return result


def get_base_config(dataset, use_baseline):
    return CONFIG_GROUPS[get_config_group(dataset)]["base_config"](dataset, use_baseline)


def get_model_config(dataset, model, use_baseline):
    group = CONFIG_GROUPS[get_config_group(dataset)]
    return group["model_configs"][model](dataset, model, use_baseline)


def get_config(dataset, model, use_baseline):
    config = {
        "use_cond_affine": not use_baseline,
        "pure_cond_affine": False,
        **get_base_config(dataset, use_baseline),
        **get_model_config(dataset, model, use_baseline)
    }

    if use_baseline:
        for prefix in ["s", "t", "st"]:
            config.pop(f"{prefix}_nets", None)

        for prefix in ["p", "q"]:
            for suffix in ["", "_mu", "_sigma"]:
                config.pop(f"{prefix}{suffix}_nets", None)

        config = {
            **config,
            "num_u_channels": 0,
            "num_train_elbo_samples": 1,
            "num_valid_elbo_samples": 1,
            "num_test_elbo_samples": 1,
        }

    return config


def expand_grid_generator(config):
    if not config:
        yield {}
        return

    items = list(config.items())
    first_key, first_val = items[0]
    rest = dict(items[1:])

    for config in expand_grid_generator(rest):
        if isinstance(first_val, GridParams):
            for val in first_val:
                yield {
                    first_key: val,
                    **config
                }

        elif isinstance(first_val, dict):
            for sub_config in expand_grid_generator(first_val):
                yield {
                    first_key: sub_config,
                    **config
                }

        else:
            yield {
                first_key: first_val,
                **config
            }


def expand_grid(config):
    return list(expand_grid_generator(config))
