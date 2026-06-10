from baselines.evaluators.evaluate_raster import (
    _component_result,
    evaluate_vector_row,
)


def assert_close(actual, expected, eps=1e-12):
    assert abs(actual - expected) <= eps, (actual, expected)


absolute = ("elevation", "absolute", ("elevation",), 10.0, "m")
angular = ("aspect", "angular", ("aspect",), 5.0, "degrees")
relative = ("count", "relative", ("count",), 0.05, "relative")
location = ("location", "location", ("geometry",), 5.0, "m")
name = ("name", "token_f1", ("name",), 0.8, "f1")

assert _component_result({"elevation": 109}, {"elevation": 100}, absolute)["passed"]
assert _component_result({"aspect": 359}, {"aspect": 1}, angular)["passed"]
assert _component_result({"count": 0}, {"count": 0}, relative)["passed"]
assert not _component_result({"count": 1}, {"count": 0}, relative)["passed"]
assert _component_result({"geometry": "POINT (-76 39)"}, {"geometry": "POINT (-76 39)"}, location)["passed"]
assert not _component_result({"geometry": "POINT (-76 39)"}, {"name": "Main Street"}, name)["passed"]

assert evaluate_vector_row({"type": "angle", "angle_error": 5 / 180 - 1e-12})["correct"]
assert not evaluate_vector_row({"type": "angle", "angle_error": 5 / 180 + 1e-12})["correct"]
assert evaluate_vector_row({"type": "loc", "distance_error": 5 / 500000 - 1e-12})["correct"]
assert not evaluate_vector_row({"type": "loc", "distance_error": 5 / 500000 + 1e-12})["correct"]

print("ok")
