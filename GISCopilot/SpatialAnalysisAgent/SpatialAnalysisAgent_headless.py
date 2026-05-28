import ast
import json
import os
import re
import sys
import time
import uuid
import requests
import urllib.parse

import SpatialAnalysisAgent_Constants as constants
import SpatialAnalysisAgent_helper as helper
import SpatialAnalysisAgent_ToolsDocumentation as ToolsDocumentation
import SpatialAnalysisAgent_Codebase as codebase


def ensure_qgis_python_paths():
    """
    Ensure QGIS Python paths are on sys.path so generated code can import qgis/processing.
    """
    prefix = os.environ.get("QGIS_PREFIX_PATH", "/Applications/QGIS.app")
    if prefix.endswith(".app"):
        resources_root = os.path.join(prefix, "Contents", "Resources")
    else:
        resources_root = prefix
    candidates = [
        os.path.join(resources_root, "python"),
        os.path.join(resources_root, "python", "plugins"),
        os.path.join(resources_root, "qgis", "python"),
    ]
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    return value_str in {"1", "true", "yes", "y", "on"}


def ensure_qgis_initialized(prefix_path=None, add_processing_provider=True):
    """
    Initialize QGIS in headless mode if it is not already initialized.
    Safe to call multiple times.
    """
    try:
        from qgis.core import QgsApplication
    except Exception:
        return None

    app = QgsApplication.instance()
    if app is None:
        # Avoid GUI requirements when running headless
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if prefix_path:
            QgsApplication.setPrefixPath(prefix_path, True)
        app = QgsApplication([], False)
        app.initQgis()

    if add_processing_provider:
        try:
            import processing  # noqa: F401
            from qgis.analysis import QgsNativeAlgorithms
            QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
        except Exception:
            # Processing provider may already be registered or unavailable.
            pass

    # Refresh algorithm registry snapshots after initialization
    try:
        codebase.refresh_processing_algorithms()
    except Exception:
        pass

    return app


def shutdown_qgis(app):
    if app is not None:
        try:
            app.exitQgis()
        except Exception:
            pass


def _normalize_data_paths(data_path):
    if isinstance(data_path, (list, tuple)):
        return [p for p in data_path if str(p).strip()]
    if data_path is None:
        return []
    # Accept semicolon or newline separated input
    raw = str(data_path)
    if ";" in raw:
        parts = raw.split(";")
    else:
        parts = raw.splitlines()
    return [p.strip() for p in parts if p.strip()]


_FORBIDDEN_QGIS_RE = re.compile(
    r"(^\s*(from\s+qgis\b|import\s+qgis\b|import\s+processing\b|from\s+processing\b))|(\bprocessing\.run\s*\()",
    re.IGNORECASE | re.MULTILINE,
)


def _has_forbidden_qgis_usage(code: str) -> bool:
    return bool(code) and bool(_FORBIDDEN_QGIS_RE.search(code))


def run_pipeline(
    task,
    data_path,
    workspace_directory,
    model_name,
    task_name_override=None,
    is_review=True,
    reasoning_effort_value="medium",
    stream=True,
    init_qgis=True,
    qgis_prefix_path=None,
):
    """
    Headless pipeline for generating and executing QGIS analysis code.
    Returns the final generated code string.
    """
    ensure_qgis_python_paths()
    if init_qgis:
        ensure_qgis_initialized(prefix_path=qgis_prefix_path)

    # Normalize inputs
    data_path_list = _normalize_data_paths(data_path)
    is_review = _coerce_bool(is_review)

    print("=" * 56)
    print(f"User: {task}")
    print("=" * 56)
    time.sleep(1)

    # Initialize the AI model
    API_Key = helper.get_openai_key(model_name)
    if 'gibd-services' in (API_Key or ''):
        request_id = helper.get_question_id(API_Key)
        print(f"RequestID:{request_id}")
    else:
        request_id = str(uuid.uuid4())

    _ = helper.initialize_ai_model(
        model_name=model_name,
        reasoning_effort=reasoning_effort_value,
        OpenAI_key=API_Key,
    )

    # Analyze the user request
    print("=" * 56)
    print("AI IS ANALYZING THE TASK ...")
    print("=" * 56)

    operation_model = helper.get_model_for_operation(model_name)
    if task_name_override:
        task_name = task_name_override
    else:
        task_name = helper.generate_task_name_with_model_provider(
            request_id=request_id,
            model_name=operation_model,
            stream=False,
            task_description=task,
            reasoning_effort=reasoning_effort_value,
        )
    print(f"task_name: {task_name}")

    # Data overview
    print("=" * 56)
    print("AI IS EXAMINING THE DATA ...")
    print("=" * 56)

    attributes_json, data_overview = helper.add_data_overview_to_data_location(
        request_id=request_id,
        task=task,
        data_location_list=data_path_list,
        model_name=operation_model,
        reasoning_effort=reasoning_effort_value,
    )
    print(f"data overview: {data_overview}")
    print(attributes_json)

    # Fine tuning user request
    query_tuning_prompt_str = helper.create_Query_tuning_prompt(
        task=task,
        data_overview=data_overview,
    )

    print(query_tuning_prompt_str)
    print("TASK_BREAKDOWN:", end="")
    task_breakdown = helper.Query_tuning(
        request_id=request_id,
        Query_tuning_prompt_str=query_tuning_prompt_str,
        model_name=model_name,
        stream=stream,
        reasoning_effort=reasoning_effort_value,
    )
    print("\n_")

    # Tool selection
    print("=" * 56)
    print("AI IS SELECTING THE APPROPRIATE TOOL(S) ...")
    print("=" * 56)

    tool_select_prompt_str = helper.create_ToolSelect_prompt(
        task=task_breakdown,
        data_path=data_overview,
    )

    print(f"TOOL SELECT PROMPT ---------------------: {tool_select_prompt_str}")
    print("SELECTED TOOLS:", end="")
    selected_tools_reply = helper.tool_select(
        request_id=request_id,
        ToolSelect_prompt_str=tool_select_prompt_str,
        model_name=operation_model,
        stream=stream,
        reasoning_effort=reasoning_effort_value,
    )
    refined_selected_tools_reply = helper.extract_dictionary_from_response(
        response=selected_tools_reply
    )

    import ast

    try:
        selected_tools_dict = ast.literal_eval(refined_selected_tools_reply)
        print(f"\nSELECTED TOOLS: {selected_tools_dict}\n")
    except (SyntaxError, ValueError) as e:
        print("Error parsing the dictionary:", e)
        selected_tools_dict = {"Selected tool": []}

    selected_tools = selected_tools_dict.get("Selected tool", [])
    if isinstance(selected_tools, str):
        selected_tools = [selected_tools]

    tools_documentation_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Tools_Documentation"
    )

    selected_tool_ids_list = []
    selected_tools_resolved = {}
    all_documentation = []

    for selected_tool in selected_tools:
        if selected_tool in codebase.algorithm_names:
            selected_tool_id = codebase.algorithms_dict[selected_tool]["ID"]
        elif selected_tool in constants.tool_names_lists:
            selected_tool_id = constants.CustomTools_dict[selected_tool]["ID"]
        else:
            selected_tool_id = selected_tool

        selected_tools_resolved[selected_tool] = selected_tool_id
        selected_tool_ids_list.append(selected_tool_id)

        selected_tool_file_id = re.sub(r"[ :?\\/]", "_", selected_tool_id)
        selected_tool_file_path = None

        for root, _, files in os.walk(tools_documentation_dir):
            for file in files:
                if file == f"{selected_tool_file_id}.toml":
                    selected_tool_file_path = os.path.join(root, file)
                    break
            if selected_tool_file_path:
                break

        if not selected_tool_file_path:
            print(f"Tool documentation for {selected_tool_file_id}.toml is not provided")
            continue

        print(f"TOOL_ID: {selected_tool_id}")
        print(f"Selected tool filename: {selected_tool_file_id}")

        if ToolsDocumentation.check_toml_file_for_errors(selected_tool_file_path):
            print(f"File {selected_tool_file_id} is free from errors.")
            documentation_str = ToolsDocumentation.tool_documentation_collection(
                tool_ID=selected_tool_file_id
            )
        else:
            print(f"File {selected_tool_file_id} has errors. Attempting to fix...")
            ToolsDocumentation.fix_toml_file(selected_tool_file_path)
            print(f"Retrieving documentation after fixing {selected_tool_file_id}.")
            documentation_str = ToolsDocumentation.tool_documentation_collection(
                tool_ID=selected_tool_file_id
            )

        all_documentation.append(documentation_str)

    print(f"List of selected tool IDs: {selected_tool_ids_list}")
    combined_documentation_str = "\n".join(all_documentation)
    print(combined_documentation_str)

    # Paths for per-task artifacts
    task_artifacts_dir = os.path.join(workspace_directory, "task_artifacts")
    code_dir = os.path.join(task_artifacts_dir, "generated_code")
    output_dir = os.path.join(task_artifacts_dir, "outputs")
    log_dir = os.path.join(task_artifacts_dir, "logs")
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Solution graph
    print("\n---------- AI IS GENERATING THE GEOPROCESSING WORKFLOW FOR THE TASK ----------\n")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    os.makedirs(save_dir, exist_ok=True)

    graph_file_path = os.path.join(save_dir, f"{task_name}.graphml")
    graph_response, code_for_graph, solution_graph_dict = helper.generate_graph_response(
        request_id=request_id,
        task=task,
        task_explanation=task_breakdown,
        data_path=data_overview,
        graph_file_path=graph_file_path,
        model_name=operation_model,
        stream=stream,
        execute=True,
        reasoning_effort=reasoning_effort_value,
    )

    if solution_graph_dict and solution_graph_dict.get("graph"):
        G = solution_graph_dict["graph"]
        nt = helper.show_graph(G)
        html_graph_path = os.path.join(save_dir, f"{task_name}_solution_graph.html")
        counter = 1
        while os.path.exists(html_graph_path):
            html_graph_path = os.path.join(save_dir, f"{task_name}_solution_graph_{counter}.html")
            counter += 1
        nt.save_graph(html_graph_path)
        print(f"GRAPH_SAVED:{html_graph_path}")
    else:
        print("Failed to generate or load solution graph")
        html_graph_path = ""

    # Operation prompt and code generation
    operation_prompt_str = helper.create_operation_prompt(
        task=task_breakdown,
        data_path=data_overview,
        workspace_directory=workspace_directory,
        selected_tools=selected_tool_ids_list,
        documentation_str=combined_documentation_str,
    )
    print(f"OPERATION PROMPT: {operation_prompt_str}")
    print("\n---------- AI IS GENERATING THE OPERATION CODE ----------\n")
    print("GENERATED CODE:", end="")

    llm_reply_str = helper.generate_operation_code(
        request_id=request_id,
        operation_prompt_str=operation_prompt_str,
        model_name=model_name,
        stream=stream,
        reasoning_effort=reasoning_effort_value,
    )

    print("\n -------------------------- GENERATED CODE --------------------------------------------\n")
    print("```python")
    extracted_code = helper.extract_code_from_str(llm_reply_str, task)
    print("```")
    if _has_forbidden_qgis_usage(extracted_code):
        print("Detected forbidden qgis/processing usage. Regenerating once with strict non-QGIS constraint.")
        strict_prompt = (
            operation_prompt_str
            + "\n\nCRITICAL CONSTRAINTS:\n"
            + "- Do NOT import qgis modules.\n"
            + "- Do NOT import/use processing.\n"
            + "- Use ijson streaming for large GeoJSON shards; do not load or concatenate full layers.\n"
            + "- Use shapely/pyproj/geopandas only after filtering to a small candidate set.\n"
            + "- For postal_codes shards, expand with glob (part-*.geojson); do not open wildcard literal path.\n"
            + "- Preserve the task's required JSON output schema and key names exactly.\n"
            + "- Resolve attributes as top-level first, then tagsMap fallback.\n"
            + "- If no valid result, print valid JSON with {\"answers\": []} only.\n"
        )
        llm_reply_str = helper.generate_operation_code(
            request_id=request_id,
            operation_prompt_str=strict_prompt,
            model_name=model_name,
            stream=stream,
            reasoning_effort=reasoning_effort_value,
        )
        extracted_code = helper.extract_code_from_str(llm_reply_str, task)

    if is_review:
        print("\n ----AI IS REVIEWING THE GENERATED CODE (YOU CAN DISABLE CODE REVIEW IN THE SETTINGS TAB)----", end="")
        code_review_prompt_str = helper.code_review_prompt(
            extracted_code=extracted_code,
            data_path=data_overview,
            selected_tool_dict=selected_tool_ids_list,
            workspace_directory=workspace_directory,
            documentation_str=combined_documentation_str,
        )
        review_str_llm_reply_str = helper.code_review(
            request_id=request_id,
            code_review_prompt_str=code_review_prompt_str,
            model_name=model_name,
            stream=stream,
            reasoning_effort=reasoning_effort_value,
        )

        print()
        print("\n\n")
        print("-------------------------- REVIEWED CODE --------------------------\n")
        print("```python")
        reviewed_code = helper.extract_code_from_str(review_str_llm_reply_str, task_breakdown)
        print("```")

        generated_code = reviewed_code
        if _has_forbidden_qgis_usage(generated_code):
            print("Reviewed code still uses qgis/processing. Regenerating once without QGIS modules.")
            strict_prompt = (
                operation_prompt_str
                + "\n\nCRITICAL CONSTRAINTS:\n"
                + "- Do NOT import qgis modules.\n"
                + "- Do NOT import/use processing.\n"
                + "- Use ijson streaming for large GeoJSON shards; do not load or concatenate full layers.\n"
                + "- Use shapely/pyproj/geopandas only after filtering to a small candidate set.\n"
                + "- For postal_codes shards, expand with glob (part-*.geojson); do not open wildcard literal path.\n"
                + "- Preserve the task's required JSON output schema and key names exactly.\n"
                + "- Resolve attributes as top-level first, then tagsMap fallback.\n"
                + "- If no valid result, print valid JSON with {\"answers\": []} only.\n"
            )
            llm_reply_str = helper.generate_operation_code(
                request_id=request_id,
                operation_prompt_str=strict_prompt,
                model_name=model_name,
                stream=stream,
                reasoning_effort=reasoning_effort_value,
            )
            generated_code = helper.extract_code_from_str(llm_reply_str, task)
        print("OPERATION CODE GENERATED AND REVIEWED SUCCESSFULLY")

        print("CODE_READY_URLENCODED:" + urllib.parse.quote(generated_code))

        code, output, error_collector = helper.execute_complete_program(
            request_id=request_id,
            code=generated_code,
            try_cnt=5,
            task=task,
            model_name=model_name,
            reasoning_effort_value=reasoning_effort_value,
            documentation_str=combined_documentation_str,
            data_path=data_path_list,
            workspace_directory=workspace_directory,
            review=True,
            stream=stream,
            reasoning_effort=reasoning_effort_value,
        )
    else:
        generated_code = extracted_code
        code, output, error_collector = helper.execute_complete_program(
            request_id=request_id,
            code=extracted_code,
            try_cnt=5,
            task=task,
            model_name=model_name,
            reasoning_effort_value=reasoning_effort_value,
            documentation_str=combined_documentation_str,
            data_path=data_path_list,
            workspace_directory=workspace_directory,
            stream=stream,
            review=True,
            reasoning_effort=reasoning_effort_value,
        )

    generated_code = code

    print("CODE_READY_URLENCODED2:" + urllib.parse.quote(generated_code))

    selected_tools_str = ", ".join(selected_tools) if isinstance(selected_tools, list) else str(selected_tools)
    try:
        html_graph_content = helper.read_html_graph_content(html_graph_path)
    except (FileNotFoundError, IOError) as e:
        print(f"ERROR: {e}")
        html_graph_content = ""

    if 'gibd-services' not in (API_Key or ''):
        print("Error reporting skipped (not using gibd-services API key)")

    url = f"https://www.gibd.online/api/feedback/{API_Key}"
    feedback = {
        "service_name": "GIS Copilot",
        "question_id": request_id,
        "question": task,
        "error_msg": "Collected execution errors",
        "error_traceback": str(error_collector),
        "generated_code": generated_code,
        "data_overview": data_overview,
        "task_breakdown": task_breakdown,
        "selected_tools": selected_tools_str,
        "workflow": html_graph_content,
    }

    try:
        requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=feedback,
        )
    except Exception:
        pass

    # Save generated code to disk for reproducibility
    code_path = os.path.join(code_dir, f"{task_name}.py")
    with open(code_path, "w") as f:
        f.write(generated_code)

    # Save raw output
    raw_output_path = os.path.join(output_dir, f"{task_name}.output.txt")
    with open(raw_output_path, "w") as f:
        f.write(output or "")

    # Parse output lines into JSON when possible
    json_lines = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # Normalize python dict output to strict JSON for downstream evaluation.
        try:
            obj = json.loads(line)
            print(json.dumps(obj), flush=True)
            json_lines.append(obj)
            continue
        except Exception:
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, dict):
                    print(json.dumps(obj), flush=True)
                    json_lines.append(obj)
                    continue
            except Exception:
                pass
        print(line, flush=True)

    # Attach token usage + estimated cost
    usage = helper.get_token_stats() if hasattr(helper, "get_token_stats") else {}
    # Pricing per 1M tokens (input/output) in USD
    pricing = {
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    }
    cost_by_model = {}
    if usage and isinstance(usage, dict):
        by_model = usage.get("by_model", {})
        for m, stats in by_model.items():
            rate = pricing.get(m)
            if not rate:
                continue
            prompt_tokens = stats.get("prompt_tokens", 0) or 0
            completion_tokens = stats.get("completion_tokens", 0) or 0
            cost = (prompt_tokens / 1_000_000.0) * rate["input"] + (completion_tokens / 1_000_000.0) * rate["output"]
            cost_by_model[m] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": stats.get("total_tokens", 0) or 0,
                "estimated_cost_usd": cost,
            }

    normalized_json_lines = []
    for obj in json_lines:
        if isinstance(obj, dict):
            # Enforce deterministic task identity for downstream evaluators.
            obj["task_name"] = task_name
            obj.setdefault("task", task)
            if not isinstance(obj.get("answers"), list):
                obj["answers"] = []
            obj["_usage"] = usage
            obj["_cost_estimate"] = cost_by_model
            normalized_json_lines.append(obj)
        else:
            # Preserve unexpected non-dict model outputs as explicit failed records.
            normalized_json_lines.append(
                {
                    "answers": [],
                    "task_name": task_name,
                    "task": task,
                    "error": "non_dict_json_output",
                    "raw_output_line": obj,
                    "_usage": usage,
                    "_cost_estimate": cost_by_model,
                }
            )

    # Save JSON outputs (always write at least one record)
    json_output_path = os.path.join(output_dir, f"{task_name}.output.jsonl")
    if not normalized_json_lines:
        normalized_json_lines = [{
            "answers": [],
            "task_name": task_name,
            "task": task,
            "error": str(error_collector) if error_collector else None,
            "_usage": usage,
            "_cost_estimate": cost_by_model,
        }]
    with open(json_output_path, "w") as f:
        for obj in normalized_json_lines:
            f.write(json.dumps(obj) + "\n")

    # Save a lightweight per-task log
    log_path = os.path.join(log_dir, f"{task_name}.log")
    with open(log_path, "w") as f:
        f.write(f"task: {task}\n")
        f.write(f"task_name: {task_name}\n")
        f.write(f"graph: {html_graph_path}\n")
        f.write(f"generated_code: {code_path}\n")
        f.write(f"raw_output: {raw_output_path}\n")
        f.write(f"json_output: {json_output_path}\n")
        f.write(f"errors: {error_collector}\n")

    return generated_code


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SpatialAnalysisAgent headlessly (QGIS Python required)."
    )
    parser.add_argument("--task", required=True, help="User task description")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Semicolon or newline separated data paths",
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--task-name",
        default=None,
        help="Optional task name override for deterministic artifact filenames",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Workspace directory for outputs",
    )
    parser.add_argument(
        "--review",
        default="true",
        help="Enable code review (true/false)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Reasoning effort for GPT-5 models",
    )
    parser.add_argument(
        "--no-qgis-init",
        action="store_true",
        help="Skip QGIS initialization (if already initialized)",
    )
    parser.add_argument(
        "--qgis-prefix",
        default=None,
        help="Optional QGIS prefix path for headless initialization",
    )
    args = parser.parse_args()

    run_pipeline(
        task=args.task,
        data_path=args.data_path,
        workspace_directory=args.workspace,
        model_name=args.model,
        task_name_override=args.task_name,
        is_review=args.review,
        reasoning_effort_value=args.reasoning_effort,
        stream=True,
        init_qgis=not args.no_qgis_init,
        qgis_prefix_path=args.qgis_prefix,
    )


if __name__ == "__main__":
    main()
