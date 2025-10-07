from pydantic import BaseModel, NonNegativeInt


class DataQualityReport(BaseModel):
    invalid_schema: NonNegativeInt
    invalid_timing: NonNegativeInt
    missing_value: NonNegativeInt
