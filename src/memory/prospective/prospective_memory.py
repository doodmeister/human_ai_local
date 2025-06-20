"""
Prospective Memory System for Human-AI Cognition Framework

Stores and manages future intentions, reminders, and scheduled tasks.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid

@dataclass
class ProspectiveMemoryItem:
    id: str
    description: str
    due_time: datetime
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completed_at: Optional[datetime] = None

class ProspectiveMemorySystem:
    def __init__(self):
        self.items: Dict[str, ProspectiveMemoryItem] = {}

    def add_reminder(self, description: str, due_time: datetime) -> str:
        item_id = str(uuid.uuid4())
        item = ProspectiveMemoryItem(id=item_id, description=description, due_time=due_time)
        self.items[item_id] = item
        return item_id

    def get_due_reminders(self, now: Optional[datetime] = None) -> List[ProspectiveMemoryItem]:
        now = now or datetime.now()
        return [item for item in self.items.values() if not item.completed and item.due_time <= now]

    def complete_reminder(self, item_id: str):
        item = self.items.get(item_id)
        if item and not item.completed:
            item.completed = True
            item.completed_at = datetime.now()

    def list_reminders(self, include_completed: bool = False) -> List[ProspectiveMemoryItem]:
        return [item for item in self.items.values() if include_completed or not item.completed]
