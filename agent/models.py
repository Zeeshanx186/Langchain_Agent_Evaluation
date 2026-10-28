import json
import os
import re
import subprocess
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

load_dotenv()


class AntigravityChatModel(BaseChatModel):
    """LangChain ChatModel powered by Antigravity / Gemini via agy CLI at zero API cost."""

    model_name: str = "gemini-3.8-flash-low"

    def __init__(self, model: str = "gemini-3.8-flash-low", **kwargs):
        resolved = os.getenv("ANTIGRAVITY_AGENT_MODEL", model)
        super().__init__(model_name=resolved, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "antigravity"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tools = kwargs.get("tools", [])

        formatted_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                formatted_messages.append(f"System: {m.content}")
            elif isinstance(m, HumanMessage):
                formatted_messages.append(f"User: {m.content}")
            elif isinstance(m, ToolMessage):
                formatted_messages.append(f"Tool Result ({m.name or getattr(m, 'tool_call_id', '')}): {m.content}")
            elif isinstance(m, AIMessage):
                tc_str = f" [Tool calls: {json.dumps(m.tool_calls)}]" if m.tool_calls else ""
                formatted_messages.append(f"Assistant: {m.content}{tc_str}")
            else:
                formatted_messages.append(f"{getattr(m, 'type', 'Message')}: {m.content}")

        history_str = "\n".join(formatted_messages)

        if not tools:
            prompt = f"Follow system instructions if present.\n\nCONVERSATION:\n{history_str}\n\nAssistant:"
            cmd = [
                "agy",
                "--dangerously-skip-permissions",
                "--effort", "low",
                "--model", self.model_name,
                "--input-format", "text",
                "--output-format", "json",
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
                raise RuntimeError(f"agy error: {stderr}")
            out_json = json.loads(stdout)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out_json.get("response", "").strip()))])

        tools_desc = [t if isinstance(t, dict) else convert_to_openai_tool(t) for t in tools]
        prompt = f"""You are an AI assistant executing tool calling steps.
Follow the system instructions strictly.

AVAILABLE TOOLS (JSON Schema):
{json.dumps(tools_desc, indent=2)}

CONVERSATION HISTORY:
{history_str}

CRITICAL INSTRUCTIONS:
1. If you need information or need to take an action, select the appropriate tool from AVAILABLE TOOLS and call it.
2. If you are ready to produce the final response to the user, you MUST call the "Answer" tool with parameters "answer" (your message to the user) and "conversation" (boolean, false if finished, true if expecting user follow-up).
3. Output MUST be strictly valid JSON matching this schema:
{{
  "thought": "brief reasoning",
  "tool_calls": [
    {{
      "name": "<tool_name>",
      "args": {{ ... }}
    }}
  ]
}}
Do NOT output markdown around the JSON, only the raw JSON.
"""
        cmd = [
            "agy",
            "--dangerously-skip-permissions",
            "--effort", "low",
            "--model", self.model_name,
            "--input-format", "text",
            "--output-format", "json",
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
            raise RuntimeError(f"agy error: {stderr}")

        out_json = json.loads(stdout)
        resp_text = out_json.get("response", "").strip()

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", resp_text)
        cleaned = match.group(1).strip() if match else resp_text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        data = json.loads(cleaned[start:end])

        tool_calls = []
        for idx, tc in enumerate(data.get("tool_calls", [])):
            tool_calls.append({
                "name": tc["name"],
                "args": tc.get("args", {}),
                "id": f"call_{uuid4().hex[:8]}_{idx}",
            })

        content = "" if tool_calls else data.get("thought", "")
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=content, tool_calls=tool_calls)
        )])


model = AntigravityChatModel()
