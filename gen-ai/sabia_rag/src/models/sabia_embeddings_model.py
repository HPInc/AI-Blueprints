from framework_classes.embeddings_model import EmbeddingsModel
from framework_classes.tokenizer import Tokenizer
from langchain_community.embeddings import LlamaCppEmbeddings

class SabIAEmbeddingsModel(EmbeddingsModel):
    """
    Especialização de `EmbeddingsModel`.
    Funções:
        - embed(): gera a representação de embeddings de uma string de tokens.
    """

    def __init__(self, model_path: str, tokenizer: Tokenizer, n_gpu_layers: int = 16) -> None:
        """
        Cria uma instância com um Tokenizer e configura o número de camadas da GPU.
        
        Args:
            model_path (str): Caminho para o modelo a ser carregado.
            tokenizer (Tokenizer): Uma instância de Tokenizer para tokenizar o texto.
            n_gpu_layers (int, opcional): Número de camadas da GPU. Padrão para 16.
        """
        self.model = LlamaCppEmbeddings(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  
            n_ctx=2048,
            n_batch=128,
            f16_kv=True
        )
        self.tokenizer = tokenizer  

    def embed(self, text: str) -> list[float]:
        """
        Gera a representação de embeddings de uma string de tokens.

        Args:
            text (str): Uma string de texto de tokens concatenados.
            
        Returns:
            list[float]: Uma lista de floats representando os tokens.
        """
        tokens = self.tokenizer.encode(text)  
        embeddings = self.model.embed_query(" ".join(tokens))  
        return embeddings
