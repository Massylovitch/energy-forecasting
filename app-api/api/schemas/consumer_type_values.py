from pydantic import BaseModel
from typing import List


class UniqueConsumerType(BaseModel):
    values: List[int]