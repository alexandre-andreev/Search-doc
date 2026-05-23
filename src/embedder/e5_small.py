from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
BATCH_SIZE = 32
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class E5SmallEmbedder:
    def __init__(self, device: str = "cuda", batch_size: int = BATCH_SIZE):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA недоступна. Установите PyTorch с CUDA-поддержкой: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME, device=self.device)
        return self._model

    def encode_passages(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Кодирует пассажи для индексации (с префиксом 'passage: ')."""
        model = self._ensure_loaded()
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        return model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Кодирует один поисковый запрос (с префиксом 'query: ')."""
        model = self._ensure_loaded()
        vec = model.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        return MODEL_NAME
