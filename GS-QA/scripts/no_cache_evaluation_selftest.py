from baselines.evaluators.evaluate_raster import COMPONENT_SPECS, _component_result, evaluate_question


def value_for(metric, alias):
    if metric in ("token_f1", "token_f1_set"):
        return "Main Street"
    if metric == "exact":
        return "yes"
    if metric == "location":
        return "POINT (-76 39)"
    return 100.0


tested = 0
for stem, specs in sorted(COMPONENT_SPECS.items()):
    pred_row = {}
    gold_row = {}
    for _, metric, aliases, _, _ in specs:
        alias = aliases[0]
        value = value_for(metric, alias)
        pred_row[alias] = value
        gold_row[alias] = value
    if any(metric == "token_f1_set" for _, metric, _, _, _ in specs):
        components = evaluate_question({
            "id": stem,
            "query_type": "selftest",
            "source_stem": stem,
            "answer_type": "",
            "predicted_exec": {"output": [pred_row], "error": ""},
            "gold_exec": {"output": [gold_row], "error": ""},
            "predicted_sql": "selftest",
        })["detail"]["components"]
    else:
        components = [
            _component_result(pred_row, gold_row, spec)
            for spec in specs
        ]
    assert components and all(item["passed"] for item in components), (stem, components)
    tested += 1

print("no_cache_component_specs", tested)
