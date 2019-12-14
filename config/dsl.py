CONFIG_GROUPS = {}
CURRENT_CONFIG_GROUP = None


def group(group, datasets):
    global CURRENT_CONFIG_GROUP

    assert group not in CONFIG_GROUPS, f"Already exists group `{group}'"

    for dataset in datasets:
        for group_data in CONFIG_GROUPS.values():
            assert dataset not in group_data["datasets"], \
                f"Dataset `{dataset}' already registered in group `{group}'"

    CONFIG_GROUPS[group] = {
        "datasets": datasets,
        "base_config": None,
        "model_configs": {}
    }

    CURRENT_CONFIG_GROUP = group


def base(f):
    assert CONFIG_GROUPS[CURRENT_CONFIG_GROUP]["base_config"] is None, \
        "Already exists a base config"
    CONFIG_GROUPS[CURRENT_CONFIG_GROUP]["base_config"] = f
    return f


def provides(*models):
    def store_and_return(f):
        assert CURRENT_CONFIG_GROUP is not None, "Must register a config group first"

        for m in models:
            assert m not in CONFIG_GROUPS[CURRENT_CONFIG_GROUP]["model_configs"], \
                f"Already exists model `{m}' in group `{CURRENT_CONFIG_GROUP}'"

            CONFIG_GROUPS[CURRENT_CONFIG_GROUP]["model_configs"][m] = f
        return f
    return store_and_return


class GridParams:
    def __init__(self, *values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(str(v) for v in self.values)})"
