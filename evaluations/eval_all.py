import json
import sys
from pathlib import Path
from uuid import uuid4

import mlflow
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

if __package__:
    from .dataset import dataset
    from .model import eval_model
else:
    from dataset import dataset
    from model import eval_model


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


def build_metrics():
    """Keep the thresholds and evaluator model used by the individual scripts."""
    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            model=eval_model,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            include_reason=True,
            async_mode=False,
        ),
        StepEfficiencyMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
        PlanQualityMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
        PlanAdherenceMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
    ]
    tool_metrics = [
        ToolCorrectnessMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "make sure that the actual output and the expected output have the same logical conclusion"
            ],
            threshold=0.8,
            model=eval_model,
            async_mode=False,
            verbose_mode=True,
        ),
    ]
    arg_metrics = [
        ArgumentCorrectnessMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
    ]
    return trace_metrics, tool_metrics, arg_metrics


def main():
    load_dotenv()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "agent"))
    from agent import agent
    from deepeval.test_run import global_test_run_manager

    mlflow.set_tracking_uri(f"sqlite:///{project_root / 'mlflow.db'}")
    mlflow.set_experiment("Agent Full Suite Evaluation")

    with mlflow.start_run(run_name="EXP-001-eval-all"):
        mlflow.log_params({
            "eval_model": str(eval_model),
            "dataset_size": len(dataset.goldens),
        })

        trace_metrics, tool_metrics, arg_metrics = build_metrics()
        cache_config = CacheConfig(use_cache=False, write_cache=False)
        display_config = DisplayConfig(inspect_after_run=False)
        async_config = AsyncConfig(run_async=False)
        error_config = ErrorConfig(ignore_errors=True)
        test_cases_all = []
        test_cases_args = []

        # The callback captures the trace required by the four agentic metrics.
        for golden in dataset.evals_iterator(
            metrics=trace_metrics,
            cache_config=cache_config,
            display_config=display_config,
            async_config=async_config,
            error_config=error_config,
        ):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": golden.input}]},
                config={
                    "configurable": {"thread_id": str(uuid4())},
                    "callbacks": [CallbackHandler(metrics=trace_metrics)],
                },
            )

            actual_output = response.get("structured_response")
            actual_output_str = (
                actual_output.answer
                if actual_output
                else response["messages"][-1].content
            )

            tools_called_all = []
            tools_called_args = []
            for message in response.get("messages", []):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    if tool_call["name"] != "Answer":
                        tc = ToolCall(
                            name=tool_call["name"],
                            input_parameters=tool_call["args"],
                            reasoning=actual_output_str,
                        )
                        tools_called_all.append(tc)
                        if tool_call["name"] in ["get_user_purchase_history", "calculate_right_refund"]:
                            tools_called_args.append(tc)

            test_cases_all.append(
                LLMTestCase(
                    input=golden.input,
                    actual_output=actual_output_str,
                    expected_output=golden.expected_output,
                    tools_called=tools_called_all,
                    expected_tools=golden.expected_tools,
                )
            )
            test_cases_args.append(
                LLMTestCase(
                    input=golden.input,
                    actual_output=actual_output_str,
                    expected_output=golden.expected_output,
                    tools_called=tools_called_args,
                )
            )

        # Evaluate Tool Correctness and Answer Correctness with all expected tools
        evaluate(
            test_cases=test_cases_all,
            metrics=tool_metrics,
            cache_config=cache_config,
            display_config=display_config,
            async_config=async_config,
            error_config=error_config,
            _skip_reset=True,
        )

        # Evaluate Argument Correctness with user-input parameterized tools
        evaluate(
            test_cases=test_cases_args,
            metrics=arg_metrics,
            cache_config=cache_config,
            display_config=display_config,
            async_config=async_config,
            error_config=error_config,
            _skip_reset=True,
        )
        print(f"\nCompleted all 7 metrics for {len(test_cases_all)} test cases.")

        # Log evaluation results to MLflow
        test_run = global_test_run_manager.get_test_run()
        test_run.construct_metrics_scores()

        if test_run.metrics_scores:
            total_passes = sum(m.passes for m in test_run.metrics_scores)
            total_fails = sum(m.fails for m in test_run.metrics_scores)
            total_evals = total_passes + total_fails
            pass_rate = (total_passes / total_evals) if total_evals > 0 else 0.0

            mlflow.log_metrics({
                "pass_rate": pass_rate,
                "passed": total_passes,
                "failed": total_fails,
                "total": total_evals,
            })

            for m in test_run.metrics_scores:
                metric_name = m.metric.lower().replace(" ", "_").replace("[", "").replace("]", "")
                if m.scores:
                    mlflow.log_metric(f"avg_{metric_name}", sum(m.scores) / len(m.scores))

            print("\n" + "=" * 70)
            print("                 FULL SUITE AGGREGATE RESULTS")
            print("=" * 70)
            for m in test_run.metrics_scores:
                avg = sum(m.scores) / len(m.scores) if m.scores else 0.0
                rate = (m.passes / len(m.scores) * 100) if m.scores else 0.0
                print(f"  {m.metric:<30} | Avg: {avg:.2f} | Pass: {rate:6.2f}% ({m.passes}/{len(m.scores)})")
            print("=" * 70)

            print("\n[MLflow] Results logged to MLflow successfully!")


if __name__ == "__main__":
    main()