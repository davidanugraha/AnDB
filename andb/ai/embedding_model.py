from andb.ai.utils import *

class EmbeddingModelFactory:
    # singleton pattern
    __embedding_model = None
    __embedding_model_type = None
    
    @classmethod
    def create_model(cls, model_type, **config):
        """
        Factory method to initialize the correct EmbeddingModel subclass based on config.

        Args:
            model_type (str): The type of model to use, defaults to openai
            config (dict): Configuration containing model type and related settings.

        Returns:
            EmbeddingModel: An instance of the appropriate subclass.
        """
        config = ModelConfig(config)
        # if user change model type, we need to reinit a new model
        if cls.__embedding_model and cls.__embedding_model_type == model_type:
            return cls.__embedding_model

        if model_type == "hf_api":
            cls.__embedding_model = HFAPIEmbeddingModel(config)
        elif model_type == "openai":
            cls.__embedding_model = OpenAIEmbeddingModel(config)
        elif model_type == "offline":
            cls.__embedding_model = OfflineEmbeddingModel(config)
        else:
            raise ValueError("Invalid model type. Choose 'hf_api', 'openai', or 'offline'.")

        cls.__embedding_model_type = model_type
        return cls.__embedding_model

class EmbeddingModel:
    def generate_embeddings(self, text):
        """
        Generate embeddings for the given text using the specified model.

        Args:
            text (str): The input text to encode into embeddings.

        Returns:
            list[float]: The generated embeddings as a list of floats.
        """
        raise NotImplementedError("generate_embeddings must be implemented in subclasses.")

class HFAPIEmbeddingModel(EmbeddingModel):
    def __init__(self, config):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(api_key=config.get("hf_api_key") or os.getenv('HF_API_KEY'))
        self.model = config.get("embed_hf_repo_id")

    def generate_embeddings(self, text):
        response = self.client.feature_extraction(text, model=self.model)
        return response

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, config):
        import openai
        from openai import OpenAI

        openai.api_key = config.get("openai_api_key")
        self.openai_client = OpenAI(api_key=config.get("openai_api_key") or os.getenv('OPENAI_API_KEY'))
        self.openai_model = config.get("embed_openai_model")

    def generate_embeddings(self, text):
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.openai_model
        )
        return response.data[0].embedding

class OfflineEmbeddingModel(EmbeddingModel):
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        import torch

        self.model = SentenceTransformer(
            config.get("embed_offline_model_path"),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    def generate_embeddings(self, text):
        embeddings = self.model.encode(text)
        return embeddings.tolist()