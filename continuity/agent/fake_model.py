"""A ``BaseLlm`` substitute for driving the pipeline in tests with zero network access.

``google.adk.models.base_llm.BaseLlm`` is the ADK-supported extension point for
swapping the model: ``LlmAgent.model`` is typed ``Union[str, BaseLlm]``, so an
``LlmAgent`` built with a ``FakeLlm`` instance is indistinguishable, from ADK's
point of view, from one built with the real Gemini model -- it never imports or
calls ``google.genai``'s network client. This is what lets
``tests/agent/test_agents.py`` drive the real ``Workflow`` pipeline,
through the real ADK function-calling and ``output_schema`` validation flow,
without ever making a model call.

``FakeLlm`` returns pre-scripted ``LlmResponse``s in order, one per call to
``generate_content_async`` -- typically one function-call turn per tool call
the script wants the "model" to make, then a final text turn whose text is
valid JSON for the agent's ``output_schema``. It records every ``LlmRequest``
it is asked to answer, so a test can assert what conversation state the agent
actually built (e.g. that a tool's result reached the next turn).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.genai import types
from pydantic import Field, PrivateAttr


def scripted_function_calls(*calls: tuple[str, Mapping[str, Any]]) -> LlmResponse:
    """A model turn that calls one or more tools by name, with the given arguments.

    `calls` is `(tool_name, arguments)` pairs. Multiple pairs script parallel
    function calls in one turn, matching what a real model turn can do.
    """
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=dict(args)))
        for name, args in calls
    ]
    return LlmResponse(content=types.Content(role="model", parts=parts))


def scripted_final_text(text: str) -> LlmResponse:
    """A model turn with no function calls -- the final, schema-validated answer."""
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        finish_reason=types.FinishReason.STOP,
    )


class FakeLlm(BaseLlm):
    """Replays `responses` in order; never touches the network.

    Construct one per stage under test (each ``LlmAgent`` gets its own model
    instance) and script exactly the turns that stage's fake "reasoning"
    should take: zero or more `scripted_function_calls(...)`, then exactly one
    `scripted_final_text(...)` whose text is valid JSON for that stage's
    `output_schema`.

    Raises `AssertionError` -- not a silent no-op -- if the agent asks for more
    turns than were scripted; a wiring bug should fail loudly in a test, not
    hang or return an empty response.
    """

    responses: list[LlmResponse] = Field(default_factory=list)

    _requests: list[LlmRequest] = PrivateAttr(default_factory=list)
    _next_index: int = PrivateAttr(default=0)

    @property
    def requests(self) -> Sequence[LlmRequest]:
        """Every `LlmRequest` this fake was asked to answer, in order."""
        return tuple(self._requests)

    @property
    def call_count(self) -> int:
        return len(self._requests)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        self._requests.append(llm_request)
        if self._next_index >= len(self.responses):
            raise AssertionError(
                f"FakeLlm(model={self.model!r}) received call "
                f"{self._next_index + 1}, but was only scripted with "
                f"{len(self.responses)} response(s). The agent asked for more "
                "reasoning turns than the test scripted."
            )
        response = self.responses[self._next_index]
        self._next_index += 1
        yield response
