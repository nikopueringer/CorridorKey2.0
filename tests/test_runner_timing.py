from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from gui.clips import DONE, FAILED, STOPPED, Clip
from gui.clip_list import CLIP_ROLE, STATUS_COLUMN, ClipList, _timing_line
from gui.runner import QueueRunner


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


class Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_live_keying_elapsed_and_eta_wait_for_two_completed_windows(app, tmp_path):
    runner = QueueRunner()
    clock = Clock()
    runner._clock = clock
    clip = Clip(tmp_path / "clip.mov", tmp_path / "clip.mov", False)
    runner._clip = clip
    runner._clip_started_at = 0.0
    runner._keying_started_at = 0.0
    runner._keying_since = 0.0
    runner._proc = object()  # enable the live-timer update path
    updates = []
    runner.clip_changed.connect(lambda item: updates.append(item.detail))
    try:
        clock.advance(10)
        runner._handle("      window 1/5 [0:48]")
        assert "0:10 elapsed" in clip.detail
        assert "left" not in clip.detail  # one completed window is too little evidence

        clock.advance(10)
        runner._handle("      window 2/5 [24:72]")
        assert "0:20 elapsed" in clip.detail
        assert "~0:30 left" in clip.detail

        clock.advance(1)
        runner._tick_timing()
        assert "0:21 elapsed" in updates[-1]
        assert "~0:29 left" in updates[-1]

        clock.advance(4)
        runner._handle("      window 5/5 [96:120]")
        assert clip.detail.startswith("Writing outputs ·")
        assert "left" not in clip.detail  # output writing has no reliable ETA
    finally:
        runner._timing_timer.stop()


def test_low_vram_stage_two_eta_and_keying_end_marker(app, tmp_path):
    runner = QueueRunner()
    clock = Clock()
    runner._clock = clock
    clip = Clip(tmp_path / "low-vram.mov", tmp_path / "low-vram.mov", False)
    runner._clip = clip
    runner._clip_started_at = 0.0
    runner._keying_started_at = 0.0
    runner._low_vram = True
    runner._proc = object()
    try:
        clock.advance(10)
        runner._handle("      window 4/4 [72:96]")
        assert runner._keying_ended_at is None
        runner._handle("    Stage2 rendering: frames=100")
        clock.advance(10)
        runner._handle("      output frames 20/100")
        assert "Stage 2 rendering 20/100" in clip.detail
        assert "left" not in clip.detail
        clock.advance(10)
        runner._handle("      output frames 40/100")
        assert "~0:30 left" in clip.detail
        clock.advance(2)
        runner._tick_timing()
        assert "~0:28 left" in clip.detail
        runner._handle("    Keying complete; closing outputs")
        assert runner._keying_ended_at == clock.now
        assert clip.detail.startswith("Writing outputs ·")
        assert "left" not in clip.detail
    finally:
        runner._timing_timer.stop()


@pytest.mark.parametrize("status", [DONE, FAILED, STOPPED])
def test_final_timing_records_outcome_total_and_keying_duration(app, tmp_path, status):
    runner = QueueRunner()
    clock = Clock(160.0)
    runner._clock = clock
    clip = Clip(tmp_path / f"{status}.mov", tmp_path / f"{status}.mov", False)
    runner._clip_started_at = 100.0
    runner._keying_started_at = 110.0
    runner._keying_ended_at = 150.0

    runner._finish_clip_timing(clip, status)

    assert clip.timing_summary == f"{status} · total 1:00 · keying 0:40"


def test_queue_total_timing_resets_for_each_run(app):
    runner = QueueRunner()
    clock = Clock(10.0)
    runner._clock = clock
    runner._run_started_at = 10.0
    runner.last_run_timing = "stale previous timing"

    clock.advance(75)
    runner._finish_run()

    assert runner.last_run_timing == "Queue total 1:15"
    assert runner._run_started_at is None


def test_timing_summaries_are_logged_for_clip_and_queue(app, tmp_path):
    runner = QueueRunner()
    clock = Clock(20.0)
    runner._clock = clock
    runner._clip_started_at = 0.0
    runner._keying_started_at = 5.0
    runner._keying_ended_at = 15.0
    clip = Clip(tmp_path / "logged.mov", tmp_path / "logged.mov", False)
    messages = []
    runner.log.connect(messages.append)

    runner._finish_clip_timing(clip, DONE)
    runner._run_started_at = 0.0
    runner._finish_run()

    assert messages == ["Timing logged: Done · total 0:20 · keying 0:10", "Queue total 0:20"]


def test_completed_clip_timing_is_visible_in_status_row(app, tmp_path):
    clip = Clip(tmp_path / "visible.mov", tmp_path / "visible.mov", False)
    clip.status = DONE
    clip.timing_summary = "Done · total 1:24 · keying 1:02"
    clip_list = ClipList()
    clip_list.add([clip])
    item = clip_list.topLevelItem(0)
    index = clip_list.model().index(0, STATUS_COLUMN)
    delegate = clip_list.itemDelegateForColumn(STATUS_COLUMN)
    from PySide6.QtWidgets import QStyleOptionViewItem
    option = QStyleOptionViewItem()
    with_timing = delegate.sizeHint(option, index).height()
    clip.timing_summary = ""
    without_timing = delegate.sizeHint(option, index).height()

    assert item.data(0, CLIP_ROLE) is clip
    assert "total 1:24 · keying 1:02" in item.toolTip(STATUS_COLUMN)
    assert with_timing == without_timing  # fixed row heights reserve space for every row
    assert _timing_line("Done · total 1:24 · keying 1:02") == "Key 1:02 · Total 1:24"
    clip_list.close()
