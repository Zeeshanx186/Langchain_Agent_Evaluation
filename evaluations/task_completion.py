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

from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler

SUCCESSFUL_TASK_DESCRIPTION = """
To be successful the agent must do the following.

if the user doesn't say what the username is: the task would be successful just by asking him to provide the username

if the user provide the username and ask for a refund:

the agent must check few things(the agent do not need to communicate to the user that it is doing these checks unless no refund will be given for this reason):

1- the purchase was made 30 days prior to today's date

2- the refund should only be for the items demanded by the user. If the user demanded all of their purchased items (such as both the book and the pen), refunding all requested items is expected and correct.

3- The refund should only be given for the items that are bought.

once you have all the data needed inform the user about the status of his refund ticket by saying one of these 2(only if a refund was requested):

    a- a refund was issued
    b- a refund was not issued. and explain why

if the user has requests that are not about a refund (such as wanting to know what they purchased):
the task is successful simply by answering the question without issuing any refund tickets. No date checks or refund denial explanations are required.
"""

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("Agent Task Completion")

with mlflow.start_run(run_name="EXP-003-flagship-eval-model"):
    mlflow.log_params({
        "eval_model": eval_model,
        "threshold": 0.8,
        "dataset_size": len(dataset.goldens),
        "changes": "Upgraded eval_model to glm-5.3 (eliminates judge hallucinations)",
    })
    mlflow.log_text(SUCCESSFUL_TASK_DESCRIPTION, "rubric.txt")

    metrics = [TaskCompletionMetric(threshold=0.8, model=eval_model, task=SUCCESSFUL_TASK_DESCRIPTION, include_reason=True, async_mode=False)]

    for golden in dataset.evals_iterator(
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
    ):
        response = agent.invoke(
            {"messages": [{"role": "user", "content": golden.input}]},
            config={
                "configurable": {"thread_id": str(uuid4())},
                "callbacks": [CallbackHandler(metrics=metrics)],
            },
        )

    # Log evaluation results to MLflow
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
                mlflow.log_metric("avg_task_completion", avg_score)

        mlflow.log_artifact(str(results_path), artifact_path="deepeval_logs")
        print("\n[MLflow] Results logged to MLflow successfully!")
    