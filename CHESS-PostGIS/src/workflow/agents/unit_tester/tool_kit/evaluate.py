from typing import Dict, List
import re

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from workflow.agents.tool import Tool

class Evaluate(Tool):
    """
    Tool for evaluating the generated unit tests.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.scores = []
        self.comparison_matrix = []
        self.SQL_id = None
        

    def _run(self, state: SystemState):
        """
        Executes the unit test evaluation process.
        
        Args:
            state (SystemState): The current system state.
        """
        try:
            key_to_evaluate = list(state.SQL_meta_infos.keys())[-1]
            target_SQL_meta_infos = state.SQL_meta_infos[key_to_evaluate]
            target_SQL_meta_infos = [q for q in target_SQL_meta_infos if self._is_valid_candidate_sql(getattr(q, "SQL", ""))]
            self._question_text = f"{state.task.question or ''} {state.task.evidence or ''}".lower()
        except Exception as e:
            print(f"Error in UnitTestEvaluator: {e}")
            return
        if key_to_evaluate.startswith(self.tool_name):
            id = int(key_to_evaluate[len(self.tool_name)+1:])
            self.SQL_id = self.tool_name + "_" + str(id+1)
        else:
            self.SQL_id = self.tool_name + "_1"  
        state.SQL_meta_infos[self.SQL_id] = []
        request_list = []
        if len(target_SQL_meta_infos) == 0:
            self.scores = []
            self.comparison_matrix = []
            return
        if len(target_SQL_meta_infos) == 1:
            state.SQL_meta_infos[self.SQL_id].append(target_SQL_meta_infos[0])
            self.scores = [1]
            self.comparison_matrix = [[1]]
            return
        if len(state.unit_tests["unit_test_generation"]) == 0:
            state.SQL_meta_infos[self.SQL_id].append(target_SQL_meta_infos[0])
            self.scores = [1]
            self.comparison_matrix = [[1]]
            return
        candidates_clusters = self.execution_based_clustering(target_SQL_meta_infos)
        formatted_candidates = ""
        for index, candidate_query in enumerate(target_SQL_meta_infos):
            formatted_candidates += f"Candidate Response #{index+1}: Query: {candidate_query.SQL}\n, Execution Result: {self._format_sql_query_result(candidate_query)}\n"
        try:
            database_schema = state.get_database_schema_for_queries(
                    [sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos]
                )
        except Exception:
            # Fall back to full schema when one candidate uses invalid columns.
            database_schema = state.get_schema_string(schema_type="complete")
        for index, unit_test in enumerate(state.unit_tests["unit_test_generation"]): 
            try:            
                request_kwargs = {
                    "DATABASE_SCHEMA": database_schema,
                    "QUESTION": state.task.question,
                    "HINT": state.task.evidence,
                    "CANDIDATE_RESPONSES": formatted_candidates,
                    "UNIT_TEST": unit_test
                }
                request_list.append(request_kwargs)
            except Exception as e:
                print(f"Error in UnitTestEvaluator while creating request list: {e}")
                continue
                
        try:
            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=request_list,
                step=self.tool_name
            )
            response = [r[0] for r in response]
        except Exception as e:
            print(f"Error in Checker while getting response: {e}")
            response = []
        comparison_matrix = []
        for item in response:
            if isinstance(item, dict) and isinstance(item.get("scores"), list):
                comparison_matrix.append(item["scores"])
        if len(comparison_matrix) == 0:
            state.SQL_meta_infos[self.SQL_id].append(target_SQL_meta_infos[0])
            self.scores = [1]
            self.comparison_matrix = [[1]]
            return
        # Sum scores across all unit tests.
        # Some parsed responses may contain fewer verdicts than candidate SQLs.
        # Normalize each score row to candidate_count to avoid index errors.
        candidate_count = len(target_SQL_meta_infos)
        normalized_matrix = []
        for score_row in comparison_matrix:
            row = list(score_row[:candidate_count])
            if len(row) < candidate_count:
                row.extend([0] * (candidate_count - len(row)))
            normalized_matrix.append(row)

        self.comparison_matrix = normalized_matrix
        scores = [
            sum(score_row[index] for score_row in normalized_matrix)
            for index in range(candidate_count)
        ]
        self.scores = scores
        # find the best candidate
        best_candidate = self.pick_the_best_candidate(scores, target_SQL_meta_infos, candidates_clusters)
        state.SQL_meta_infos[self.SQL_id].append(best_candidate)


    def self_consistency(self, candidate_clusters: Dict) -> SQLMetaInfo:
        """
        picks the candidate with the largest cluster.
        
        Args:
            candidates_clusters (Dict): The clusters of the candidates.
        """
        largest_cluster = max(candidate_clusters, key=lambda x: len(candidate_clusters[x]))
        return candidate_clusters[largest_cluster][0]
    

    def test_case_filtering_based_on_inter_cluster_variance(
            self,
            candidates_clusters: Dict,
            scores: List[int],
            target_SQL_meta_infos: List[SQLMetaInfo]
    ) -> bool:
        """
        Filters the test cases based on the inter-cluster variance.
        
        Args:
            candidates_clusters (Dict): The clusters of the candidates.
            scores (List[int]): The scores of the candidates.
            target_SQL_meta_infos (List[SQLMetaInfo]): The target SQL meta information.
        """
        for key, candidates in candidates_clusters.items():
            cluster_scores = [scores[target_SQL_meta_infos.index(candidate)] for candidate in candidates]
            if len(set(cluster_scores)) > 1:
                return False
        return True

    def pick_the_best_candidate(self, scores: List[int], candidates: List[SQLMetaInfo], candidate_clusters: Dict) -> SQLMetaInfo:
        """
        Picks the best candidate based on the scores.
        
        Args:
            scores (List[int]): The scores of the candidates.
            candidates (List[SQLMetaInfo]): The candidates.
            candidate_clusters (Dict): The clusters of the candidates.
        """
        largest_cluster = max(candidate_clusters, key=lambda x: len(candidate_clusters[x]))
        # Spatial tie-break for spatial-intent questions: prefer highest-scoring spatial SQL.
        if self._is_spatial_question():
            spatial_indices = [
                i for i, candidate in enumerate(candidates)
                if self._spatial_sql_score(getattr(candidate, "SQL", "")) > 0
            ]
            if spatial_indices:
                spatial_scores = [scores[i] for i in spatial_indices]
                best_spatial_score = max(spatial_scores)
                best_spatial_candidates = [candidates[i] for i in spatial_indices if scores[i] == best_spatial_score]
                if len(best_spatial_candidates) == 1:
                    return best_spatial_candidates[0]
                for candidate in best_spatial_candidates:
                    if candidate in candidate_clusters[largest_cluster]:
                        return candidate
                return best_spatial_candidates[0]
        max_score = max(scores)
        best_candidates = [candidates[index] for index, score in enumerate(scores) if score == max_score]
        if len(best_candidates) == 1:
            return best_candidates[0]
        for candidate in best_candidates:
            if candidate in candidate_clusters[largest_cluster]:
                return candidate
        return best_candidates[0]

    def _is_spatial_question(self) -> bool:
        text = getattr(self, "_question_text", "")
        spatial_markers = [
            "nearest", "closest", "nearby", "near ", "vicinity", "distance", "within",
            "kilometer", "km", "meter", "miles", "range", "direction", "north", "south",
            "east", "west", "course", "angle", "azimuth", "location",
        ]
        return any(marker in text for marker in spatial_markers)

    def _spatial_sql_score(self, sql: str) -> int:
        s = (sql or "").lower()
        score = 0
        if "st_distance" in s:
            score += 3
        if "geomfromwkb(" in s:
            score += 2
        if "geometry" in s:
            score += 1
        # Penalize known bad geometry usage patterns.
        if re.search(r"st_distance\s*\(\s*[^,]*osm_id", s):
            score -= 3
        if "geomfromwkb(t1.osm_id" in s or "geomfromwkb(osm_id" in s:
            score -= 3
        if "<distance_threshold>" in s or "<some_distance>" in s:
            score -= 2
        return score

    def _is_valid_candidate_sql(self, sql: str) -> bool:
        s = (sql or "").strip()
        if not s:
            return False
        if re.match(r"^(select|with|pragma)\b", s, flags=re.IGNORECASE) is None:
            return False
        if "<" in s or ">" in s:
            return False
        return True
    

    def _format_sql_query_result(self, sql_meta_info: SQLMetaInfo) -> str:
        """
        Formats the SQL query to pass to the picker model.
        
        Args:
            sql_meta_info (SQLMetaInfo): The SQL meta information.
        """
        try:
            execution_result = sql_meta_info.execution_result
            if execution_result is None:
                return "No results"
            if not isinstance(execution_result, list):
                execution_result = list(execution_result)
            number_of_rows = len(execution_result)
            if number_of_rows == 0:
                number_of_columns = 0
            else:
                number_of_columns = len(execution_result[0])
            if number_of_rows > 20:
                execution_result = execution_result[:20]
            formatted_result = (
                f"Rows: {number_of_rows}, Columns: {number_of_columns}, Results:"
                f" {execution_result}"
            )
        except Exception as e:
            formatted_result = f"Error: {e}"
        return formatted_result
    
    def execution_based_clustering(self, candidate_queries: List[SQLMetaInfo]) -> list:
        """
        Clusters the generated candidates based on the execution results.
        
        Args:
            state (SystemState): The current system state.
        """
        clusters = {}
        for query in candidate_queries:
            try:
                result = str(query.execution_result) if isinstance(query.execution_result, str) else repr(query.execution_result)
            except Exception:
                continue
            if result not in clusters:
                clusters[result] = []
            clusters[result].append(query)
        # sample one query from each cluster
        return clusters
        
    def _get_updates(self, state: SystemState) -> Dict:
        keys = list(state.SQL_meta_infos.keys())
        key_to_evaluate = keys[-2] if len(keys) >= 2 else (keys[-1] if keys else "")
        target_SQL_meta_infos = state.SQL_meta_infos.get(key_to_evaluate, [])
        selected_sql = "--"
        if self.SQL_id and self.SQL_id in state.SQL_meta_infos and len(state.SQL_meta_infos[self.SQL_id]) > 0:
            selected_sql = state.SQL_meta_infos[self.SQL_id][0].SQL
        return {
            "scores": self.scores,
            "comparison_matrix": self.comparison_matrix,
            "candidates": [sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos],
            "selected_candidate": selected_sql
        }
