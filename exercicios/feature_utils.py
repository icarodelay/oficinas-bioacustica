"""Utilitário autossuficiente para explorar o cache de features do GlicoVoz.

Depende apenas de `pandas` (+ `pyarrow` para ler parquet) — sem `parselmouth`, `librosa`,
`torch`/`transformers` nem o pacote `glicovoz`. Os parquets de cache (gerados por
`scripts/extract_features.py`, um arquivo por paciente, ex. `P1R_F25.parquet`) já vêm
junto, em `notebooks/outputs/features/`. Isso faz da pasta `notebooks/` inteira — código +
dados — uma unidade autossuficiente: basta copiar/exportar `notebooks/` sozinha para outro
lugar que o notebook continua funcionando.

Cada parquet de paciente contém metadados (glicemia, horário, etc.) + as 3 famílias de
features já extraídas: Praat (`praat_*`), espectral/timbre via librosa (`librosa_*`) e
Wav2Vec2 (`wav2vec_*`). `get_patient_features` apenas lê e filtra essas colunas — não
reextrai nada a partir de áudio.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Prefixo de coluna de cada técnica no parquet de cache.
FEATURE_TECHNIQUES = {
    "praat": "praat_",
    "spectral": "librosa_",
    "wav2vec": "wav2vec_",
}

# Colunas de metadado esperadas em todo parquet de cache (mesmo schema mínimo usado
# pelo pipeline de extração do projeto).
METADATA_COLUMNS = [
    "sample_id",
    "patient_id",
    "age",
    "day",
    "month",
    "hour",
    "minute",
    "day_key",
    "day_ordinal",
    "glycemia",
    "glycemia_class",
    "filepath",
]

# outputs/features/ dentro de notebooks/ — tudo contido nesta pasta, sem depender do
# resto do repositório.
DEFAULT_FEATURES_DIR = Path(__file__).resolve().parent / "outputs" / "features"


def list_available_patients(features_dir: Path | None = None) -> list[str]:
    """Lista os IDs de paciente com cache disponível (um `.parquet` por paciente)."""
    features_dir = features_dir or DEFAULT_FEATURES_DIR
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Pasta de cache não encontrada: {features_dir}. Verifique se "
            "`notebooks/outputs/features/` contém os parquets (ou passe `features_dir=` "
            "explicitamente)."
        )
    return sorted(p.stem for p in features_dir.glob("*.parquet"))


def get_patient_features(
    patient_id: str,
    technique: str,
    features_dir: Path | None = None,
) -> pd.DataFrame:
    """Retorna metadados + features de uma única técnica para um paciente.

    Args:
        patient_id: ID do paciente (nome do parquet sem extensão, ex. "P1R_F25").
        technique: uma das chaves de `FEATURE_TECHNIQUES` — "praat", "spectral" ou "wav2vec".
        features_dir: pasta com os parquets de cache (default: `notebooks/outputs/features/`).

    Returns:
        DataFrame com as colunas de metadados (`sample_id`, `patient_id`, `glycemia`, ...)
        mais as colunas de feature da técnica pedida (prefixo `praat_`/`librosa_`/`wav2vec_`).
    """
    if technique not in FEATURE_TECHNIQUES:
        raise ValueError(f"Técnica {technique!r} desconhecida. Use uma de: {list(FEATURE_TECHNIQUES)}")

    features_dir = features_dir or DEFAULT_FEATURES_DIR
    available = list_available_patients(features_dir)
    if patient_id not in available:
        raise ValueError(f"Paciente {patient_id!r} sem cache em {features_dir}. Disponíveis: {available}")

    df = pd.read_parquet(features_dir / f"{patient_id}.parquet")
    missing_metadata = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing_metadata:
        raise ValueError(f"Parquet de {patient_id!r} não tem o schema esperado, faltam: {missing_metadata}")

    prefix = FEATURE_TECHNIQUES[technique]
    keep_cols = [c for c in df.columns if c in METADATA_COLUMNS or c.startswith(prefix)]
    return df[keep_cols].reset_index(drop=True)
