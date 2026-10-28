import os
import json
import sys
from pathlib import Path
from uuid import uuid4

import mlflow

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
os.environ["DEEPEVAL_NO_INSPECT_PROMPT"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from agent import agent
from dataset import dataset
from model import eval_model

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.integrations.langchain import CallbackHandler


mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("Agent Tool Correctness")

with mlflow.start_run(run_name="EXP-001-tool-correctness"):
    mlflow.log_params({
        "eval_model": eval_model,
        "threshold": 0.8,
        "dataset_size": len(dataset.goldens),
        "changes": "Initial tool correctness evaluation using glm-5.3",
    })

    # Step 1: Run agent on goldens and collect test cases with tools_called & expected_tools
    print("Collecting agent responses and tool calls...")
    dataset.test_cases = []
    for golden in dataset.goldens:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": golden.input}]},
            config={
                "configurable": {"thread_id": str(uuid4())},
                "callbacks": [CallbackHandler()],
            },
        )

        actual_output = response.get("structured_response")
        actual_output_str = actual_output.answer if actual_output else response["messages"][-1].content

        tools_called = []
        for message in response.get("messages", []):
            for tool_call in getattr(message, "tool_calls", []) or []:
                if tool_call["name"] != "Answer":
                    tools_called.append(
                        ToolCall(name=tool_call["name"], input_parameters=tool_call["args"])
                    )

        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output_str,
            tools_called=tools_called,
            expected_tools=golden.expected_tools,
        )
        dataset.add_test_case(test_case)

    # Step 2: Evaluate test cases with ToolCorrectnessMetric
    metrics = [
        ToolCorrectnessMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        )
    ]

    evaluate(
        test_cases=dataset.test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
    )

    # Step 3: Log evaluation results to MLflow
    results_path = Path(__file__).parent / ".deepeval" / ".latest_test_run.json"
    if not results_path.exists():
        results_path = PROJECT_ROOT / ".deepeval" / ".latest_test_run.json"

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f).get("testRunData", {})

        passed = data.get("testPassed", 0)
        failed = data.get("testFailed", 0)
        total = passed + failed
        pass_rate = (passed / total) if total > 0 else 0.0

        mlflow.log_metrics({
            "pass_rate": pass_rate,
            "passed": passed,
            "failed": failed,
            "total": total,
        })

        for m in data.get("metricsScores", []):
            scores = m.get("scores", [])
            if scores:
                avg_score = sum(scores) / len(scores)
                mlflow.log_metric("avg_tool_correctness", avg_score)

        mlflow.log_artifact(str(results_path), artifact_path="deepeval_logs")
        print("\n[MLflow] Results logged to MLflow successfully!")
