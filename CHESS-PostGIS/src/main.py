import argparse
import yaml
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from runner.run_manager import RunManager

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use.")
    parser.add_argument('--task_timeout_seconds', type=int, default=0, help="Maximum seconds to spend on one task. Use 0 to disable.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    parser.add_argument('--pick_final_sql', type=bool, default=False, help="Pick the final SQL from the generated SQLs.")
    parser.add_argument('--sql_dialect', type=str, default='sqlite', help="SQL dialect to target: sqlite, postgres, or postgis.")
    parser.add_argument('--pg_database', type=str, default=None, help="PostgreSQL database name for postgres/postgis mode.")
    parser.add_argument('--pg_host', type=str, default=None, help="PostgreSQL host for postgres/postgis mode.")
    parser.add_argument('--pg_user', type=str, default=None, help="PostgreSQL user for postgres/postgis mode.")
    parser.add_argument('--pg_password', type=str, default=None, help="PostgreSQL password for postgres/postgis mode.")
    parser.add_argument('--pg_port', type=int, default=None, help="PostgreSQL port for postgres/postgis mode.")
    args = parser.parse_args()

    args.run_start_time = datetime.now().isoformat()
    with open(args.config, 'r') as file:
        args.config=yaml.safe_load(file)
    
    return args

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()
    os.environ["CHESS_SQL_DIALECT"] = args.sql_dialect
    if args.pg_database is not None:
        os.environ["CHESS_PG_DATABASE"] = args.pg_database
    if args.pg_host is not None:
        os.environ["CHESS_PG_HOST"] = args.pg_host
    if args.pg_user is not None:
        os.environ["CHESS_PG_USER"] = args.pg_user
    if args.pg_password is not None:
        os.environ["CHESS_PG_PASSWORD"] = args.pg_password
    if args.pg_port is not None:
        os.environ["CHESS_PG_PORT"] = str(args.pg_port)
    dataset = load_dataset(args.data_path)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()

if __name__ == '__main__':
    main()
