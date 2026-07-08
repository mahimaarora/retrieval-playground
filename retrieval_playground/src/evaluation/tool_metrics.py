"""
Tool and agent evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from retrieval_playground.src.evaluation.base import mean_score


@dataclass
class ToolTrace:
    query: str
    expected_tool: str
    actual_tool: str
    expected_method: Optional[str] = None
    actual_method: Optional[str] = None
    success: Optional[bool] = None


@dataclass
class ToolEvalResult:
    scores: Dict[str, float]
    details: Dict[str, List[Dict]]


class ToolEvaluator:
    """Evaluate tool selection and routing in agentic workflows."""

    def evaluate(self, traces: List[ToolTrace]) -> ToolEvalResult:
        tool_matches = [float(t.actual_tool == t.expected_tool) for t in traces]

        method_traces = [t for t in traces if t.expected_method is not None]
        method_matches = [
            float(t.actual_method == t.expected_method) for t in method_traces
        ]

        success_traces = [t for t in traces if t.success is not None]
        success_rates = [float(t.success) for t in success_traces]

        return ToolEvalResult(
            scores={
                "tool_selection_accuracy": mean_score(tool_matches),
                "retrieval_method_accuracy": mean_score(method_matches),
                "tool_call_success_rate": mean_score(success_rates),
            },
            details={
                "tool_selection_accuracy": [
                    {
                        "query": t.query,
                        "expected": t.expected_tool,
                        "actual": t.actual_tool,
                        "match": t.actual_tool == t.expected_tool,
                    }
                    for t in traces
                ]
            },
        )

    @staticmethod
    def build_traces_from_routing(
        cases: List[Dict[str, str]],
        route_fn,
    ) -> List[ToolTrace]:
        traces = []
        for case in cases:
            decision = route_fn(case["query"], return_metadata=True)
            traces.append(
                ToolTrace(
                    query=case["query"],
                    expected_tool=case["expected_tool"],
                    actual_tool=decision.get("tool", "unknown"),
                    expected_method=case.get("expected_method"),
                    actual_method=decision.get("retrieval_method"),
                    success=decision.get("requires_retrieval", True),
                )
            )
        return traces
