import threading
from pathlib import Path
from typing import Optional
import os

# 本项目中的BGE模型使用PyTorch，避免Transformers误加载不兼容的Keras 3后端。
os.environ.setdefault("USE_TF", "0")

from langchain_openai import ChatOpenAI
from openai import OpenAI
from knowledge.core.settings import get_settings
from knowledge.utils.clients.base import BaseClientManager, logger
from FlagEmbedding import BGEM3FlagModel, FlagReranker


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_text_client: Optional[ChatOpenAI] = None
    _openai_llm_text_lock = threading.Lock()

    _openai_llm_json_client: Optional[ChatOpenAI] = None
    _openai_llm_json_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3FlagModel] = None
    _bge_m3_lock = threading.Lock()

    _bge_reranker_client: Optional[FlagReranker] = None
    _bge_reranker_lock = threading.Lock()

    # ── VLM ──

    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            settings = get_settings()
            base_url, api_key = settings.require(
                "dashscope_api_base",
                "dashscope_api_key",
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("DashScope VLM 客户端初始化成功 (base_url=%s)", base_url)

            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── LLM ──
    @classmethod
    def get_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        if response_format:
            return cls._get_or_create("_openai_llm_json_client", cls._openai_llm_json_lock,
                                      lambda: cls._create_llm_client(response_format))
        else:
            return cls._get_or_create("_openai_llm_text_client", cls._openai_llm_text_lock,
                                      lambda: cls._create_llm_client(response_format))

    @classmethod
    def _create_llm_client(cls, response_format) -> ChatOpenAI:
        try:
            settings = get_settings()
            base_url, api_key, model_name = settings.require(
                "dashscope_api_base",
                "dashscope_api_key",
                "llm_default_model",
            )

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            llm_client = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                openai_api_key=api_key,
                openai_api_base=base_url,
                model_kwargs=model_kwargs
            )
            logger.info("DashScope LLM 客户端初始化成功 (model=%s)", model_name)
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── BGE-M3嵌入模型客户端 ──
    @classmethod
    def get_bge_m3_client(cls) -> BGEM3FlagModel:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3FlagModel:
        """
        创建bge_m3 客户端
        Returns:
        """

        try:
            settings = get_settings()
            model_name = settings.require("bge_m3_path")[0]
            device = settings.bge_device
            fp16 = settings.bge_fp16 and not device.lower().startswith("cpu")
            # 2. 创建
            bge_m3_ef = BGEM3FlagModel(
                model_name_or_path=model_name,
                devices=device,
                use_fp16=fp16
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败: {e}") from e

    # ── BGE重排模型客户端 ──
    @classmethod
    def get_bge_reranker_client(cls) -> FlagReranker:
        """获取BGE重排模型，同一进程内只创建一次。"""
        return cls._get_or_create(
            "_bge_reranker_client",
            cls._bge_reranker_lock,
            cls._create_bge_reranker_client,
        )

    @classmethod
    def _create_bge_reranker_client(cls) -> FlagReranker:
        """从本地目录加载BGE重排模型。"""
        try:
            settings = get_settings()
            model_path = settings.require("bge_reranker_path")[0]
            device = settings.bge_reranker_device

            if not Path(model_path).is_dir():
                raise EnvironmentError(f"BGE重排模型目录不存在: {model_path}")

            use_fp16 = settings.bge_reranker_fp16

            # CPU环境不使用FP16，避免模型初始化或推理失败。
            if device.lower().startswith("cpu"):
                use_fp16 = False

            reranker = FlagReranker(
                model_name_or_path=model_path,
                devices=device,
                use_fp16=use_fp16,
                max_length=512,
            )

            logger.info(
                "BGE重排模型初始化成功 "
                f"(path={model_path}, device={device}, fp16={use_fp16})"
            )
            return reranker

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE重排模型初始化失败: {e}")
            raise ConnectionError(f"BGE重排模型初始化失败: {e}") from e
