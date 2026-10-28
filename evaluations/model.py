import json
import os
import re
import subprocess
import sys
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()
os.environ.setdefault("DEEPEVAL_NO_INSPECT_PROMPT", "1")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")


class AntigravityJudgeModel(DeepEvalBaseLLM):
    """DeepEval LLM judge powered by Antigravity / Gemini via agy CLI at zero API cost."""

    def __init__(self, model: str = "gemini-3.8-flash-low"):
        self.model_name = os.getenv("ANTIGRAVITY_EVAL_MODEL", model)
        super().__init__(model=self.model_name)

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return self.model_name

    def __str__(self) -> str:
        return self.model_name

    def __repr__(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema=None, **kwargs):
        cmd = [
            "agy",
            "--dangerously-skip-permissions",
            "--effort",
            "low",
            "--model",
            self.model_name,
            "--input-format",
            "text",
            "--output-format",
            "json",
        ]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout, stderr = p.communicate(input=prompt)
        if p.returncode != 0:
            raise RuntimeError(f"agy evaluation error: {stderr}")

        start = stdout.find("{")
        end = stdout.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError(f"Invalid output from agy: {stdout}")

        out_json = json.loads(stdout[start:end])
        raw_text = out_json.get("response", "").strip()

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        cleaned = match.group(1).strip() if match else raw_text

        if schema is not None:
            try:
                parsed = json.loads(cleaned)
                return schema.model_validate(parsed)
            except Exception:
                return cleaned

        return cleaned

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)


eval_model = AntigravityJudgeModel()

