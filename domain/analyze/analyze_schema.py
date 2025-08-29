from pydantic import BaseModel

class AnalyzeResult(BaseModel):
    toy_type: str
    battery: str
    material: str
    damage: str
    donate: bool
    donate_reason: str
    repair_or_disassemble: str
    token_usage: dict