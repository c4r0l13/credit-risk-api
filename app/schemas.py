"""Request/response contracts. Validation happens here, before the model sees anything."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PayStatus = Annotated[int, Field(ge=-2, le=9, description="-2..0 = no delay, k = k months late")]
Amount = Annotated[float, Field(ge=-1e7, le=1e8)]


class Application(BaseModel):
    """Six months of account history for one applicant (most recent month first)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "application_id": "app-000123",
                "LIMIT_BAL": 50000,
                "EDUCATION": 2,
                "pay_status": [2, 2, 0, 0, 0, 0],
                "bill_amt": [46990, 48233, 49291, 28314, 28959, 29547],
                "pay_amt": [0, 2000, 1500, 1000, 1000, 800],
            }
        },
    )

    application_id: str = Field(min_length=1, max_length=64)
    LIMIT_BAL: float = Field(gt=0, le=1e8, description="Credit limit")
    EDUCATION: int = Field(ge=0, le=6)
    pay_status: list[PayStatus] = Field(min_length=6, max_length=6)
    bill_amt: list[Amount] = Field(min_length=6, max_length=6)
    pay_amt: list[Annotated[float, Field(ge=0, le=1e8)]] = Field(min_length=6, max_length=6)

    def to_raw(self) -> dict[str, float]:
        raw: dict[str, float] = {"LIMIT_BAL": self.LIMIT_BAL, "EDUCATION": self.EDUCATION}
        for i, name in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
            raw[name] = self.pay_status[i]
        for i in range(6):
            raw[f"BILL_AMT{i + 1}"] = self.bill_amt[i]
            raw[f"PAY_AMT{i + 1}"] = self.pay_amt[i]
        return raw


class Decision(BaseModel):
    application_id: str
    probability_of_default: float = Field(ge=0, le=1)
    decision: Literal["approve", "decline"]
    reasons: list[str] = Field(description="Top factors increasing risk (adverse-action reasons)")
    model_version: str


class BatchRequest(BaseModel):
    applications: list[Application] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    decisions: list[Decision]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str


class DriftReport(BaseModel):
    window_size: int
    psi: dict[str, float]
    status: dict[str, Literal["stable", "moderate", "significant", "insufficient_data"]]
