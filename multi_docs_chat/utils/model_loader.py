import os
import sys
import json

from dotenv import load_dotenv

from multi_docs_chat.logger import GLOBAL_LOGGER as log
from multi_docs_chat.exceptions.custom_exception import DocumentPortalException
from multi_docs_chat.utils.config_loader import load_config

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

class ApiKeyManager:
    REQUIRED_KEYS = ["OPENAI_API_KEY"]

    def __init__(self):
        self.api_keys = {}

        raw = os.getenv("apikeyliveclass")

        if raw:
            try:
                parsed = json.loads(raw)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "apikeyliveclass is not a valid JSON object"
                    )

                self.api_keys = parsed

                log.info(
                    "Loaded API keys from ECS secret"
                )

            except Exception as e:
                log.warning(
                    "Failed to parse API keys JSON",
                    error=str(e)
                )

        for key in self.REQUIRED_KEYS:

            if not self.api_keys.get(key):

                env_val = os.getenv(key)

                if env_val:
                    self.api_keys[key] = env_val

                    log.info(
                        f"Loaded {key} from environment variable"
                    )

        missing = [
            key
            for key in self.REQUIRED_KEYS
            if not self.api_keys.get(key)
        ]

        if missing:
            log.error(
                "Missing required API keys",
                missing_keys=missing
            )

            raise DocumentPortalException(
                "Missing API keys",
                sys
            )

        log.info(
            "API keys loaded successfully",
            keys={
                k: v[:6] + "..."
                for k, v in self.api_keys.items()
            }
        )

    def get(self, key: str) -> str:

        value = self.api_keys.get(key)

        if not value:
            raise KeyError(
                f"API key for {key} is missing"
            )

        return value


class ModelLoader:
    """
    Loads embedding models and LLMs based on config.yaml
    """

    def __init__(self):

        if os.getenv(
            "ENV",
            "local"
        ).lower() != "production":

            load_dotenv()

            log.info(
                "Running in LOCAL mode: .env loaded"
            )

        else:

            log.info(
                "Running in PRODUCTION mode"
            )

        self.api_key_mgr = ApiKeyManager()

        self.config = load_config()

        log.info(
            "YAML config loaded",
            config_keys=list(self.config.keys())
        )

    def load_embeddings(self):
        """
        Load OpenAI Embeddings
        """

        try:

            model_name = self.config[
                "embedding_model"
            ]["model_name"]

            log.info(
                "Loading embedding model",
                model=model_name
            )

            return OpenAIEmbeddings(
                model=model_name,
                api_key=self.api_key_mgr.get(
                    "OPENAI_API_KEY"
                )
            )

        except Exception as e:

            log.error(
                "Error loading embedding model",
                error=str(e)
            )

            raise DocumentPortalException(
                "Failed to load embedding model",
                sys
            )

    def load_llm(self):
        """
        Load configured LLM
        """

        try:

            llm_block = self.config["llm"]

            provider_key = os.getenv(
                "LLM_PROVIDER",
                "openai"
            )

            if provider_key not in llm_block:

                log.error(
                    "LLM provider not found in config",
                    provider=provider_key
                )

                raise ValueError(
                    f"LLM provider '{provider_key}' not found in config"
                )

            llm_config = llm_block[
                provider_key
            ]

            provider = llm_config.get(
                "provider"
            )

            model_name = llm_config.get(
                "model_name"
            )

            temperature = llm_config.get(
                "temperature",
                0
            )

            max_tokens = llm_config.get(
                "max_output_tokens",
                2048
            )

            log.info(
                "Loading LLM",
                provider=provider,
                model=model_name
            )

            if provider == "openai":

                return ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get(
                        "OPENAI_API_KEY"
                    ),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            else:

                log.error(
                    "Unsupported LLM provider",
                    provider=provider
                )

                raise ValueError(
                    f"Unsupported LLM provider: {provider}"
                )

        except Exception as e:

            log.error(
                "Error loading LLM",
                error=str(e)
            )

            raise DocumentPortalException(
                "Failed to load LLM",
                sys
            )


if __name__ == "__main__":

    loader = ModelLoader()

    print("\nLoading Embeddings...")
    embeddings = loader.load_embeddings()

    print(
        f"Embedding Model Loaded: {embeddings}"
    )

    embedding_result = embeddings.embed_query(
        "Hello, how are you?"
    )

    print(
        f"Embedding Vector Length: {len(embedding_result)}"
    )

    print("\nLoading LLM...")
    llm = loader.load_llm()

    print(
        f"LLM Loaded: {llm}"
    )

    response = llm.invoke(
        "Hello, how are you?"
    )

    print(
        f"LLM Response: {response.content}"
    )