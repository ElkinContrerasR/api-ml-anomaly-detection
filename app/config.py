from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    model_store_path: str = "./models_store"
    data_path: str = "./data"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
        }


settings = Settings()
