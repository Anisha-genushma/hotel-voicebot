from pydantic import BaseModel, Field
from typing import List


class TranscribeResponse(BaseModel):
    transcript: str = Field(..., description="Full transcript of the uploaded audio, start to finish")
    duration_seconds: float = Field(..., description="Audio duration in seconds, read from the file")


class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Full call transcript text")
    duration_seconds: float = Field(..., gt=0, description="Total call duration in seconds")
    topics: List[str] = Field(
        default_factory=lambda: ["rooms", "food", "amenities", "pricing", "location", "other"],
        description="Topic buckets to classify discussion time against",
    )


class TopicBreakdown(BaseModel):
    topic: str
    estimated_seconds: float
    percentage_of_call: float
    summary: str


class AnalyzeResponse(BaseModel):
    call_duration_seconds: float
    overall_sentiment: str
    sentiment_score: float  # -1.0 (very negative) to 1.0 (very positive)
    topic_breakdown: List[TopicBreakdown]
    booking_probability: float  # 0.0 to 1.0
    key_signals: List[str]
    reasoning: str