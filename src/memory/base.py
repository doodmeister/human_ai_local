from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence

class BaseMemorySystem(ABC):
    """
    Abstract base class for all memory systems (STM, LTM, Episodic, Semantic).
    Defines a unified interface for storing, retrieving, deleting, and searching memories or facts.
    """

    @abstractmethod
    def store(self, *args, **kwargs) -> Union[str, bool]:
        """
        Store a memory or fact. Returns a unique ID or True/False.
        """
        pass

    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[Union[dict, tuple]]:
        """
        Retrieve a memory or fact by ID. Returns a dict/tuple or None.
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory or fact by ID. Returns True if deleted.
        """
        pass

    @abstractmethod
    def search(self, query: Optional[str] = None, **kwargs) -> Sequence[dict | tuple]:
        """
        Search for memories or facts. Returns a sequence of results.
        """
        pass
