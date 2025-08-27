import json
from pathlib import Path
from typing import List

class MemoryStore:
    def __init__(self, path="memory.jsonl"):
        self.path=Path(path)
        self.path.touch(exist_ok=True)
    def add_note(self,user_id:str, note:str, tags:List[str] = None):
        rec={"user_id":user_id,"note":note, "tags":tags or []}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def get_notes(self, user_id:str, limit:int=8):
        lines=self.path.read_text(encoding="utf-8").strip().splitlines()
        notes=[json.loads(l) for l in lines if l]
        # newest first: take last N for simplicity
        sel=[n["note"] for n in notes if n["user_id"]==user_id]
        return sel[-limit:]
