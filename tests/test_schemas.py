from pathlib import Path
import json

from config import get_config, get_schema


def load_schema(name):
    with open(Path("tests") / "schemas" / f"{name}.json", "r") as f:
        return json.load(f)


def test_baseline_multiscale_realnvp_schema():
    config = get_config(dataset="mnist", model="realnvp", use_baseline=True)
    schema = get_schema(config)
    true_schema = load_schema("realnvp_schema")
    assert schema == true_schema


def test_lgf_multiscale_realnvp_schema():
    config = get_config(dataset="mnist", model="realnvp", use_baseline=False)
    schema = get_schema(config)
    true_schema = load_schema("lgf_realnvp_schema")
    assert schema == true_schema


def test_baseline_glow_schema():
    config = get_config(dataset="cifar10", model="glow", use_baseline=True)
    schema = get_schema(config)
    true_schema = load_schema("glow_schema")
    assert schema == true_schema


def test_lgf_glow_schema():
    config = get_config(dataset="cifar10", model="glow", use_baseline=False)
    schema = get_schema(config)
    true_schema = load_schema("lgf_glow_schema")
    assert schema == true_schema
