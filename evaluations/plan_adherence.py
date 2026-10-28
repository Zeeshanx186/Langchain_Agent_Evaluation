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
from deepeval.metrics import PlanAdherenceMetric
from deepeval.integrations.langchain import CallbackHandler


mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("Agent Plan Adherence")

with mlflow.start_run(run_name="EXP-001-plan-adherence"):
    mlflow.log_params({
        "eval_model": eval_model,
        "threshold": 0.8,
        "dataset_size": len(dataset.goldens),
        "changes": "Upgraded eval_model to glm-5.3 (eliminates judge hallucinations)",
    })
    
    metrics = [PlanAdherenceMetric(threshold=0.8, model=eval_model, include_reason=True, async_mode=False)]

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
                mlflow.log_metric("avg_plan_adherence", avg_score)

        mlflow.log_artifact(str(results_path), artifact_path="deepeval_logs")
        print("\n[MLflow] Results logged to MLflow successfully!")