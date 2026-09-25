"""App-owned workers for user-submitted Agent turns."""

from concurrent.futures import ThreadPoolExecutor

from app.agent.service import AgentService


class AgentRunExecutor:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-run")

    def submit(self, service: AgentService, run_id: str) -> None:
        self._executor.submit(service.execute_run, run_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
