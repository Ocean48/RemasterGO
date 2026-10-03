"""Unit tests for JobQueueController."""

from core.job_queue import JobQueueController
from core.probe import StrategyConfig


def test_job_queue_addition_and_clear():
    jq = JobQueueController()
    strategy = StrategyConfig(
        input_path="sample_640x360.avi",
        output_path="out.mkv",
        target_width=1920,
        target_height=1080
    )
    job = jq.add_job("sample_640x360.avi", strategy)
    assert len(jq.jobs) == 1
    assert job.status == "Pending"
    assert job.strategy.output_path.endswith(".mkv")

    jq.clear_queue()
    assert len(jq.jobs) == 0
