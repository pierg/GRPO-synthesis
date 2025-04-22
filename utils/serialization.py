import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Type, TypeVar, get_type_hints
from pathlib import Path
import numpy as np
import torch

T = TypeVar("T")


def is_serializable(obj: Any) -> bool:
    """Check if an object is directly JSON serializable."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return True
    if isinstance(obj, (list, tuple)):
        return all(is_serializable(item) for item in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and is_serializable(v) for k, v in obj.items())
    return False


def serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable format."""
    if is_serializable(value):
        return value
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    elif is_dataclass(value):
        return serialize_dataclass(value)
    elif isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    else:
        return str(value)


def deserialize_value(value: Any, target_type: Type) -> Any:
    """Convert a serialized value back to its original type."""
    if target_type in (str, int, float, bool, type(None)):
        return target_type(value)
    elif target_type in (list, tuple):
        return target_type(
            deserialize_value(item, target_type.__args__[0]) for item in value
        )
    elif target_type == dict:
        return {
            k: deserialize_value(v, target_type.__args__[1]) for k, v in value.items()
        }
    elif target_type == np.ndarray:
        return np.array(value)
    elif target_type == torch.Tensor:
        return torch.tensor(value)
    elif is_dataclass(target_type):
        return deserialize_dataclass(value, target_type)
    else:
        return value


def serialize_dataclass(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary."""
    if not is_dataclass(obj):
        raise ValueError(f"Object {obj} is not a dataclass")

    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        result[field.name] = serialize_value(value)
    return result


def deserialize_dataclass(data: Dict[str, Any], target_class: Type[T]) -> T:
    """Convert a dictionary back to a dataclass instance."""
    if not is_dataclass(target_class):
        raise ValueError(f"Target class {target_class} is not a dataclass")

    type_hints = get_type_hints(target_class)
    kwargs = {}

    for field in fields(target_class):
        if field.name in data:
            value = data[field.name]
            field_type = type_hints.get(field.name, type(value))
            kwargs[field.name] = deserialize_value(value, field_type)

    return target_class(**kwargs)


def save_dataclass(obj: Any, filepath: Path) -> None:
    """Save a dataclass instance to a JSON file."""
    if not is_dataclass(obj):
        raise ValueError(f"Object {obj} is not a dataclass")

    serialized = serialize_dataclass(obj)
    with open(filepath, "w") as f:
        json.dump(serialized, f, indent=2)


def load_dataclass(filepath: Path, target_class: Type[T]) -> T:
    """Load a dataclass instance from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return deserialize_dataclass(data, target_class)


def save_dataclass_list(objs: list, filepath: Path) -> None:
    """Save a list of dataclass instances to a JSON file."""
    if not all(is_dataclass(obj) for obj in objs):
        raise ValueError("All objects must be dataclasses")

    serialized = [serialize_dataclass(obj) for obj in objs]
    with open(filepath, "w") as f:
        json.dump(serialized, f, indent=2)


def load_dataclass_list(filepath: Path, target_class: Type[T]) -> list[T]:
    """Load a list of dataclass instances from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [deserialize_dataclass(item, target_class) for item in data]
