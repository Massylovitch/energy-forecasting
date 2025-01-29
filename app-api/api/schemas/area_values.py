from pydantic import BaseModel
from typing import List


class UniqueArea(BaseModel):
    values: List[int]