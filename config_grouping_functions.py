import json
import glob


def get_config(path):
    with open(f"{path}/config.json", "r") as f:
        return json.load(f)


def get_configs(root):
    return [get_config(p) for p in glob.glob(f"{root}/*")]


def all_keys(configs):
    result = []
    for c in configs:
        result += list(c)
    return list(set(result))


def should_ignore_key(key):
    return key in ["seed", "schema_type", "max_bad_valid_epochs", "valid_batch_size", "test_batch_size"]


def differing_keys(root):
    configs = get_configs(root)

    result = []
    for k in all_keys(configs):
        if not should_ignore_key(k) and key_value_differs(k, configs):
            result.append(k)

    return result


def key_value_differs(key, configs):
    if key not in configs[0]:
        return True

    val = configs[0][key]

    for c in configs:
        if key not in c or c[key] != val:
            return True

    return False


def get_config_values(keys, path):
    config = get_config(path)

    result = {}
    for k in keys:
        val = config.get(k)
        result[k] = simplify_value(val)

    return result


def simplify_value(val):
    if isinstance(val, list):
        items = set(val)
        if len(items) == 1:
            item, = items
            return f"[{item}]*{len(val)}"
        else:
            return f"[{','.join(items)}]"
    elif val is None:
        return "None"
    else:
        return val


def group_runs(root):
    keys = differing_keys(root)

    groups = {}
    for run in glob.glob(f"{root}/*"):
        differing_config_dict = get_config_values(keys, run)
        vals = tuple([differing_config_dict[k] for k in keys])
        groups.setdefault(vals, []).append(run)

    return list(groups.values())
