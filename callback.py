from multiprocessing.pool import MapResult
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
class AgentCallbackHandler (BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print(f"****** \n Prompt to LLM was: \n {prompts[0]}  \n ******")
   
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print(f"****** \n LLM response: \n {response.generations[0][0].text}  \n ******")