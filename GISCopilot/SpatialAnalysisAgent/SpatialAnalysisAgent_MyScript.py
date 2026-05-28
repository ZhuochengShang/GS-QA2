#%% Import packages
import os
import sys

#%% Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
SpatialAnalysisAgent_dir = os.path.join(current_script_dir, "SpatialAnalysisAgent")
if SpatialAnalysisAgent_dir not in sys.path:
    sys.path.append(SpatialAnalysisAgent_dir)

from SpatialAnalysisAgent_headless import run_pipeline


def _get_arg(name, argv_idx=None, default=None):
    if name in globals():
        return globals()[name]
    if argv_idx is not None and len(sys.argv) > argv_idx:
        return sys.argv[argv_idx]
    return default


task = _get_arg("task", 1)
data_path = _get_arg("data_path", 2)
model_name = _get_arg("model_name", 3)
workspace_directory = _get_arg("workspace_directory", 4)
is_review = _get_arg("is_review", 5, True)
reasoning_effort_value = _get_arg("reasoning_effort_value", 6, "medium")

if task is None or data_path is None or model_name is None or workspace_directory is None:
    raise ValueError("Missing required parameters: task, data_path, model_name, workspace_directory.")

generated_code = run_pipeline(
    task=task,
    data_path=data_path,
    workspace_directory=workspace_directory,
    model_name=model_name,
    is_review=is_review,
    reasoning_effort_value=reasoning_effort_value,
    stream=True,
    init_qgis=True,
)
