from framework_classes.completion_model import CompletionModel
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class DemoModel(CompletionModel):

    def __init__(self, **kwargs) -> None:
        """
            Creates an instance with the parameters:
            - n_gpu_layers (int): default 64
            - n_batch (int): default 512
            - n_ctx (int): default 4096
            - max_tokens (int): default 1024
            - stop_tags (list[str]): default []
        """
        n_gpu_layers = kwargs.get("n_gpu_layers", 64)
        n_batch = kwargs.get("n_batch", 512)
        n_ctx = kwargs.get("n_ctx", 4096)
        max_tokens = kwargs.get("max_tokens", 1024)
        stop_tags = kwargs.get("stop_tags", [])
        model_path = kwargs.get("model_path", "ggml-model-f16-Q5_K_M.gguf")
        print("model_path")

        super().__init__()
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            f16_kv=True,  
            callback_manager=callback_manager,
            verbose=False,
            stop=stop_tags,
            streaming=False,
            temperature=0.4,
        )
    
    def predict(self, message: str) -> str:
        return self.llm(message, echo=True)