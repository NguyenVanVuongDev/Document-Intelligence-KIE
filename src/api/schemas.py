from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReceiptStatus = Literal["processed", "pending_review", "reconciled", "rejected"]


class ReceiptUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merchant: str | None = Field(default=None, max_length=255)
    date: Date | None = None
    total: str | None = Field(default=None, max_length=32)
    address: str | None = None
    status: ReceiptStatus | None = None