# -*- coding: utf-8 -*-
"""Platform-neutral dataset discovery, loading, and analysis.

The primary API in this module is :func:`resolve_dataset`.  It resolves a
natural-language dataset description against either MySQL or PostgreSQL and
returns a JSON-safe :class:`DatasetResolution`.  Applying that resolution to a
workflow node is deliberately left to the selected platform Action.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from embedding_models.embedding_model import Embedding_Model
from llm.llm import Qwen_Model
from prompts.data_loader_prompts import (
    EDA_RESULT_TEMPLATE,
    RETR_DATASET_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from utils.logs import logger
from utils.utils import (
    get_data_info_path,
    load_json,
    update_data_info,
)


SUPPORTED_DATABASE_BACKENDS = ("mysql", "postgresql")
DEFAULT_RETRIEVER = "all-mpnet-base-v2"
MAX_LLM_ATTEMPTS = 3


class DatasetResolutionError(RuntimeError):
    """Base exception for dataset resolution failures."""


class DatasetNotFoundError(DatasetResolutionError):
    """Raised when no database table can be selected for a description."""


class ColumnConfigurationError(DatasetResolutionError):
    """Raised when the LLM cannot produce a valid column configuration."""


@dataclass
class DatasetResolution:
    """Platform-neutral result consumed by a concrete platform Action."""

    data_description: str
    backend: str
    table_name: str
    columns: list[str]
    attr_mapping: dict[str, dict[str, Any]]
    column_roles: dict[str, dict[str, str]]
    column_metadata: list[dict[str, Any]]
    sample_values: dict[str, list[Any]]
    eda_result: str
    price: float
    input_tokens: int
    output_tokens: int

    def as_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-safe representation."""

        return _json_safe(asdict(self))


@dataclass
class _LLMSelection:
    table_name: str
    input_tokens: int
    output_tokens: int
    price: float


@dataclass
class _ColumnConfiguration:
    attr_mapping: dict[str, dict[str, Any]]
    column_roles: dict[str, dict[str, str]]
    column_metadata: list[dict[str, Any]]
    sample_values: dict[str, list[Any]]
    input_tokens: int
    output_tokens: int
    price: float


class _MySQLAdapter:
    """Adapt the existing MySQL helper to the resolver's table-only API."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        # Keep the MySQL connector out of Orange/PostgreSQL-only processes.
        # Platform-specific database drivers are imported only when their
        # backend is actually selected.
        from utils.mysql_utils import MySQLDatabase

        normalized = _normalize_database_config("mysql", config)
        self.database = MySQLDatabase(
            host=normalized["host"],
            port=normalized["port"],
            user=normalized["user"],
            password=normalized["password"],
            database=normalized["database"],
        )

    def connect(self) -> None:
        self.database.connect()

    def close_connection(self) -> None:
        connection = self.database.connection
        if connection is not None and connection.is_connected():
            self.database.close_connection()

    def get_database_info(self) -> list[dict[str, Any]]:
        return self.database.get_database_info()

    def read_table(self, table_name: str) -> tuple[list[tuple[Any, ...]], list[str]]:
        # The name has already been checked against discovered metadata.  It is
        # still quoted to make unusual but valid MySQL identifiers safe.
        quoted_table = "`" + table_name.replace("`", "``") + "`"
        result = self.database.read_query(f"SELECT * FROM {quoted_table}")
        if result is None:
            raise DatasetResolutionError(
                f"Failed to read MySQL table {table_name!r}"
            )
        return result


class DatasetResolver:
    """Resolve a dataset description using one configured database backend."""

    def __init__(
        self,
        backend: str,
        db_config: Mapping[str, Any],
        *,
        retriever: str = DEFAULT_RETRIEVER,
        embedding_model_factory: Callable[[str], Any] = Embedding_Model,
        llm_factory: Callable[[], Any] = Qwen_Model,
        database: Any | None = None,
    ) -> None:
        self.backend = _validate_backend(backend)
        self.db_config = _normalize_database_config(self.backend, db_config)
        self.retriever = retriever
        self.embedding_model_factory = embedding_model_factory
        self.llm_factory = llm_factory
        self.db = database or self._create_database()

    def resolve(
        self,
        data_description: str,
        *,
        user_requirement: str | None = None,
        update_context: bool = True,
    ) -> DatasetResolution:
        """Resolve, load, classify, and summarize a database table.

        Connections are always closed in ``finally``.  The returned value does
        not include database connection details or credentials.
        """

        if not isinstance(data_description, str) or not data_description.strip():
            raise ValueError("data_description must be a non-empty string")

        resolution: DatasetResolution | None = None
        try:
            self.db.connect()
            datasets = self._normalize_database_info(
                self.db.get_database_info()
            )
            selection = self._retrieve_dataset(data_description, datasets)
            allowed_tables = {
                dataset["dataset_name"] for dataset in datasets
            }
            if selection.table_name not in allowed_tables:
                raise DatasetNotFoundError(
                    "The selected table is not present in database metadata"
                )

            data = self._load_table(selection.table_name)
            requirement = (
                user_requirement
                if user_requirement is not None
                else _read_current_instruction()
            )
            column_config = self._configure_columns(data, requirement)
            eda_result = self._eda(
                data,
                column_config.column_roles,
                column_config.sample_values,
            )

            resolution = DatasetResolution(
                data_description=data_description,
                backend=self.backend,
                table_name=selection.table_name,
                columns=[str(column) for column in data.columns],
                attr_mapping=column_config.attr_mapping,
                column_roles=column_config.column_roles,
                column_metadata=column_config.column_metadata,
                sample_values=column_config.sample_values,
                eda_result=eda_result,
                price=selection.price + column_config.price,
                input_tokens=(
                    selection.input_tokens + column_config.input_tokens
                ),
                output_tokens=(
                    selection.output_tokens + column_config.output_tokens
                ),
            )
        except DatasetNotFoundError:
            if update_context:
                update_data_info(dataset_info="")
            raise
        finally:
            try:
                self.db.close_connection()
            except Exception as exc:
                logger.warning(f"关闭数据库连接失败: {exc}")

        if resolution is None:
            raise DatasetResolutionError("Dataset resolution produced no result")
        if update_context:
            update_data_info(dataset_info=resolution.eda_result)
        return resolution

    def _create_database(self) -> Any:
        if self.backend == "mysql":
            return _MySQLAdapter(self.db_config)
        from utils.postgres_utils import PostgreSQLDatabase

        return PostgreSQLDatabase(
            host=self.db_config["host"],
            port=self.db_config["port"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            database=self.db_config["database"],
            schema=self.db_config["schema"],
        )

    @staticmethod
    def _normalize_database_info(
        datasets: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(datasets, list):
            raise DatasetResolutionError(
                "Database metadata must be a list of datasets"
            )

        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for dataset in datasets:
            if not isinstance(dataset, Mapping):
                continue
            name = dataset.get("dataset_name")
            columns = dataset.get("columns")
            if (
                not isinstance(name, str)
                or not name
                or name in seen_names
                or not isinstance(columns, (list, tuple))
            ):
                continue
            normalized.append(
                {
                    "dataset_name": name,
                    "columns": [str(column) for column in columns],
                }
            )
            seen_names.add(name)

        if not normalized:
            raise DatasetNotFoundError(
                "No tables were discovered in the configured database"
            )
        return normalized

    def _retrieve_dataset(
        self,
        data_description: str,
        datasets: list[dict[str, Any]],
    ) -> _LLMSelection:
        corpus = [
            dataset["dataset_name"]
            + json.dumps(dataset["columns"], ensure_ascii=False)
            for dataset in datasets
        ]

        if len(datasets) == 1:
            recalled_datasets = datasets
        else:
            model = self.embedding_model_factory(self.retriever)
            pairs = model.get_scores(
                [data_description],
                corpus,
                topk=min(7, len(corpus)),
            )
            recalled_datasets = [
                datasets[index]
                for _, index in pairs
                if isinstance(index, (int, np.integer))
                and 0 <= int(index) < len(datasets)
            ]
            if not recalled_datasets:
                raise DatasetNotFoundError(
                    "Embedding retrieval returned no database tables"
                )

        dataset_names = [
            dataset["dataset_name"] for dataset in recalled_datasets
        ]
        logger.info(f"语义相似度检索的数据集: {dataset_names}")

        system_prompt = RETR_DATASET_SYSTEM_PROMPT.format(
            datasets=dataset_names
        )
        messages: list[Any] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_description},
        ]
        llm = self.llm_factory()
        input_tokens_total = 0
        output_tokens_total = 0
        price_total = 0.0

        for _ in range(MAX_LLM_ATTEMPTS):
            content, message, input_tokens, output_tokens, price = llm.predict(
                messages=messages
            )
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens
            price_total += price

            parsed = load_json(content) if isinstance(content, str) else {}
            selected_name = (
                parsed.get("dataset_name")
                if isinstance(parsed, Mapping)
                else None
            )
            if selected_name in dataset_names:
                return _LLMSelection(
                    table_name=selected_name,
                    input_tokens=input_tokens_total,
                    output_tokens=output_tokens_total,
                    price=price_total,
                )

            if selected_name:
                feedback = (
                    f"你选择的数据集 {selected_name!r} 不合法，"
                    f"请从以下数据集中选择：{dataset_names}"
                )
            else:
                feedback = (
                    "JSON解析错误。请返回能被 json.loads() 解析的严格 JSON，"
                    "并重新选择一个候选数据集。"
                )
            if message is not None:
                messages.append(message)
            messages.append({"role": "user", "content": feedback})

        raise DatasetNotFoundError(
            "The LLM could not select a valid database table"
        )

    def _load_table(self, table_name: str) -> pd.DataFrame:
        logger.info(f"正在从 {self.backend} 加载数据表: {table_name}")
        rows, column_names = self.db.read_table(table_name)
        return pd.DataFrame(rows, columns=column_names)

    def _configure_columns(
        self,
        data: pd.DataFrame,
        user_requirement: str,
    ) -> _ColumnConfiguration:
        columns = [str(column) for column in data.columns]
        sample_values = self._sample_values(data)
        prompt = USER_PROMPT.format(
            user_requirement=user_requirement,
            columns_info=columns,
        )
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        llm = self.llm_factory()
        input_tokens_total = 0
        output_tokens_total = 0
        price_total = 0.0
        column_roles: dict[str, dict[str, str]] | None = None

        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            content, message, input_tokens, output_tokens, price = llm.predict(
                messages=messages
            )
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens
            price_total += price

            parsed = load_json(content) if isinstance(content, str) else {}
            validation_error = _validate_column_roles(parsed, columns)
            if validation_error is None:
                column_roles = {
                    column: {
                        "role": parsed[column]["role"],
                        "type": parsed[column]["type"],
                    }
                    for column in columns
                }
                break

            logger.info(f"第 {attempt} 次列配置未通过校验")
            if message is not None:
                messages.append(message)
            messages.append(
                {
                    "role": "user",
                    "content": validation_error,
                }
            )

        if column_roles is None:
            raise ColumnConfigurationError(
                "The LLM could not produce a valid column configuration"
            )

        role_map = {
            "feature": 0,
            "target": 1,
            "meta": 2,
            "skip": -1,
        }
        type_map = {
            "numeric": 2,
            "categorical": 1,
            "text": 3,
            "datetime": 4,
        }
        attr_mapping: dict[str, dict[str, Any]] = {}
        column_metadata: list[dict[str, Any]] = []
        for index, column in enumerate(columns):
            role_info = column_roles[column]
            attr_mapping[column] = {
                "key": index,
                "name": column,
                "type": type_map[role_info["type"]],
                "role": role_map[role_info["role"]],
            }
            column_metadata.append(
                {
                    "name": column,
                    "role": role_info["role"],
                    "type": role_info["type"],
                    "sample_values": sample_values[column],
                }
            )

        return _ColumnConfiguration(
            attr_mapping=attr_mapping,
            column_roles=column_roles,
            column_metadata=column_metadata,
            sample_values=sample_values,
            input_tokens=input_tokens_total,
            output_tokens=output_tokens_total,
            price=price_total,
        )

    @staticmethod
    def _sample_values(data: pd.DataFrame) -> dict[str, list[Any]]:
        clean_data = data.dropna()
        sample_source = clean_data if len(clean_data) >= 3 else data
        sample_size = min(3, len(sample_source))
        if sample_size:
            rng = random.Random(_read_random_seed())
            sample_indexes = rng.sample(range(len(sample_source)), sample_size)
            sample_rows = sample_source.iloc[sample_indexes]
        else:
            sample_rows = sample_source.iloc[0:0]

        samples: dict[str, list[Any]] = {}
        for column in data.columns:
            samples[str(column)] = [
                _json_safe(value) for value in sample_rows[column].tolist()
            ]
        return samples

    @staticmethod
    def _eda(
        data: pd.DataFrame,
        column_roles: Mapping[str, Mapping[str, str]],
        sample_values: Mapping[str, list[Any]],
    ) -> str:
        eda_result = EDA_RESULT_TEMPLATE.replace("{shape}", str(data.shape))
        target_name = next(
            (
                column
                for column, metadata in column_roles.items()
                if metadata["role"] == "target"
            ),
            None,
        )
        if target_name is None:
            target_distribution = "No target column was identified."
        elif column_roles[target_name]["type"] == "categorical":
            target_distribution = data[target_name].value_counts().to_string()
        else:
            target_distribution = data[target_name].describe().to_string()
        eda_result = eda_result.replace(
            "{target_distribution}", target_distribution
        )

        sample_data = [
            {
                "column name": column,
                "sample data": sample_values[column],
            }
            for column, metadata in column_roles.items()
            if metadata["role"] == "feature"
        ]
        eda_result = eda_result.replace(
            "{sample_data}",
            json.dumps(sample_data, ensure_ascii=False),
        )

        missing_values = data.isnull().sum()
        missing_values = missing_values[missing_values > 0].to_string()
        return eda_result.replace(
            "{missing_value_check}", missing_values
        )


def resolve_dataset(
    data_description: str,
    backend: str,
    db_config: Mapping[str, Any],
    *,
    user_requirement: str | None = None,
    update_context: bool = True,
) -> DatasetResolution:
    """Convenience API for platform Actions."""

    resolver = DatasetResolver(backend=backend, db_config=db_config)
    return resolver.resolve(
        data_description,
        user_requirement=user_requirement,
        update_context=update_context,
    )


class Data_Loader:
    """Deprecated compatibility wrapper for the original tuple-returning API.

    New platform Actions should call :func:`resolve_dataset` and apply the
    returned resolution themselves.  ``ai_studio`` is accepted only so older
    callers can still construct this class; this neutral wrapper never imports
    or invokes AI Studio.
    """

    def __init__(
        self,
        ai_studio: Any | None = None,
        annotation: bool = False,
        *,
        backend: str = "mysql",
        db_config: Mapping[str, Any] | None = None,
    ) -> None:
        del annotation
        self.ai_studio = ai_studio
        self.backend = _validate_backend(backend)
        self.db_config = dict(
            db_config or _default_database_config(self.backend)
        )
        self.resolver = DatasetResolver(self.backend, self.db_config)
        self.last_resolution: DatasetResolution | None = None

    def run(
        self,
        data_description: str,
        node_id: str | int | None = None,
    ) -> tuple[str, float, int, int]:
        del node_id
        self.last_resolution = self.resolver.resolve(data_description)
        return (
            self.last_resolution.eda_result,
            self.last_resolution.price,
            self.last_resolution.input_tokens,
            self.last_resolution.output_tokens,
        )


def load_data(
    data_description: str,
    node_id: str | int | None = None,
    *,
    backend: str = "mysql",
    db_config: Mapping[str, Any] | None = None,
) -> tuple[str, float, int, int]:
    """Deprecated tuple-returning wrapper retained for existing imports."""

    loader = Data_Loader(
        backend=backend,
        db_config=db_config,
    )
    return loader.run(data_description, node_id)


def _validate_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise ValueError(
            f"backend must be one of {SUPPORTED_DATABASE_BACKENDS}"
        )
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_DATABASE_BACKENDS:
        raise ValueError(
            f"Unsupported database backend {backend!r}; "
            f"expected one of {SUPPORTED_DATABASE_BACKENDS}"
        )
    return normalized


def _normalize_database_config(
    backend: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("db_config must be a mapping")

    normalized = {
        "host": config.get("host", config.get("server")),
        "port": config.get("port"),
        "user": config.get("user", config.get("username")),
        "password": config.get("password"),
        "database": config.get("database"),
    }
    if backend == "postgresql":
        normalized["schema"] = config.get("schema", "public")

    missing = [
        key
        for key, value in normalized.items()
        if value is None or value == ""
    ]
    if missing:
        raise ValueError(
            f"Missing {backend} database configuration: {', '.join(missing)}"
        )
    return normalized


def _validate_column_roles(
    parsed: Any,
    columns: list[str],
) -> str | None:
    if not isinstance(parsed, Mapping):
        return (
            "JSON解析错误。请返回严格 JSON，并为所有给定列配置 role 和 type。"
        )
    if set(parsed.keys()) != set(columns):
        return (
            "请重新输入，确保配置列与给定的数据集列完全一致，"
            "不要遗漏或猜测列名。"
        )

    valid_roles = {"feature", "target", "meta", "skip"}
    valid_types = {"numeric", "categorical", "datetime", "text"}
    for column in columns:
        value = parsed.get(column)
        if not isinstance(value, Mapping):
            return f"列 {column!r} 的配置必须是包含 role 和 type 的对象。"
        if value.get("role") not in valid_roles:
            return f"列 {column!r} 的 role 不合法，请重新配置所有列。"
        if value.get("type") not in valid_types:
            return f"列 {column!r} 的 type 不合法，请重新配置所有列。"
    return None


def _default_database_config(backend: str) -> Mapping[str, Any]:
    # Kept lazy so importing the neutral resolver does not bind it to a
    # platform or expose one platform's configuration to another.
    if backend == "mysql":
        from config import MySQL_Config

        return MySQL_Config
    from config import PG_CONFIG

    return PG_CONFIG


def _read_random_seed() -> int:
    try:
        from config import RANDOM_SEED

        return int(RANDOM_SEED)
    except (ImportError, TypeError, ValueError):
        return 42


def _read_current_instruction() -> str:
    data_path = get_data_info_path()
    try:
        with open(data_path, "r", encoding="utf-8") as file:
            data_info = json.load(file)
    except (OSError, json.JSONDecodeError):
        return ""
    instruction = data_info.get("instruction", "")
    return instruction if isinstance(instruction, str) else str(instruction)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (datetime, date, time, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)
