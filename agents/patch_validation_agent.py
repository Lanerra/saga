import logging
from typing import Dict, List, Optional, Tuple

import config
from core.llm_interface import llm_service
from models import PatchInstruction, ProblemDetail
from prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


class PatchValidationAgent:
    """Validates patch instructions using an LLM."""

    def __init__(self, model_name: str = config.EVALUATION_MODEL) -> None:
        self.model_name = model_name
        logger.info("PatchValidationAgent initialized with model: %s", self.model_name)

    async def validate_patch(
        self,
        context_snippet: str,
        patch: PatchInstruction,
        problems: List[ProblemDetail],
    ) -> Tuple[bool, Optional[Dict[str, int]]]:
        """Return True if patch addresses all problems adequately."""

        issues_list = "\n".join(f"- {p['problem_description']}" for p in problems)
        prompt = render_prompt(
            "patch_validation_agent/validate_patch.j2",
            {
                "context_snippet": context_snippet,
                "patch_text": patch.get("replace_with", ""),
                "issues_list": issues_list,
            },
        )

        response_text, usage = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.Temperatures.EVALUATION,
            max_tokens=1024,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=config.FREQUENCY_PENALTY_EVALUATION,
            presence_penalty=config.PRESENCE_PENALTY_EVALUATION,
            auto_clean_response=True,
        )

        first_line = response_text.splitlines()[0].strip().lower()
        score = 0
        for token in first_line.split():
            if token.isdigit():
                score = int(token)
                break
        is_pass = score >= config.PATCH_VALIDATION_THRESHOLD
        if not is_pass:
            logger.info("Patch validation score %d below threshold", score)
        return is_pass, usage
