import httpx
import logging
from src.config import get_settings

logger = logging.getLogger(__name__)


class GuardrailsClient:
    """HTTP client for NeMo Guardrails sidecar."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.guardrails_url

    async def check_input(self, message: str) -> dict:
        """Run input rails on user message."""
        if not self.settings.use_guardrails:
            return {"safe": True, "message": message}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/rails/input",
                    json={"input": message}
                )
                return resp.json()
        except Exception as e:
            logger.warning(f'Guardrails input check failed: {e}. Passing through.')
            return {"safe": True, "message": message}

    async def check_output(self, response: str) -> dict:
        """Run output rails on LLM response."""
        if not self.settings.use_guardrails:
            return {"safe": True, "response": response}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/rails/output",
                    json={"output": response}
                )
                return resp.json()
        except Exception as e:
            logger.warning(f'Guardrails output check failed: {e}. Passing through.')
            return {"safe": True, "response": response}
