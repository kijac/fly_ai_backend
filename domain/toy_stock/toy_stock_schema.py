from pydantic import BaseModel
from typing import Optional

class Toy(BaseModel):
    toy_id: int
    toy_name: str
    toy_type: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None

    class Config:
        orm_mode = True


class ToyStockList(BaseModel):
    total: int = 0
    toystock_list: list[Toy] = []
        