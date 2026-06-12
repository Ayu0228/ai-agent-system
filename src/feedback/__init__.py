"""User feedback closed loop — collection, analysis, routing, improvement.

ref: LangChain data flywheel — production failure → annotation → regression test
ref: Anthropic — human feedback at checkpoints for continuous improvement
"""

from src.feedback.collector import FeedbackCollector, FeedbackEntry, FeedbackType
from src.feedback.analyzer import FeedbackAnalyzer, AnalysisResult, ImprovementTicket
from src.feedback.flywheel import DataFlywheel, FlywheelStage

__all__ = [
    "FeedbackCollector", "FeedbackEntry", "FeedbackType",
    "FeedbackAnalyzer", "AnalysisResult", "ImprovementTicket",
    "DataFlywheel", "FlywheelStage",
]
