"""Batching utilities for efficient database operations."""

from typing import Any, Generator, List, TypeVar

T = TypeVar("T")


def batch_iterator(items: List[T], batch_size: int = 1000) -> Generator[List[T], None, None]:
    """Yield batches of items.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def chunk_list(items: List[T], chunk_size: int) -> Generator[List[T], None, None]:
    """Split a list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Maximum size of each chunk

    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def flatten_list(nested_list: List[List[T]]) -> List[T]:
    """Flatten a nested list.

    Args:
        nested_list: Nested list structure

    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]
