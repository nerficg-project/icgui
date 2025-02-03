# -- coding: utf-8 --

"""ModelRunners/FPSRollingAverage.py: Helper class to facilitate calculation of fps rolling averages."""

from time import perf_counter

import numpy as np


class FPSRollingAverage:
    """Class to calculate the rolling average of the FPS. The FPS is calculated as the inverse of the frametime."""
    def __init__(self, window_size=10):
        self._fps_window = window_size
        self._frame_times = np.full((window_size,), np.nan, dtype=np.float32)
        self._frame_times_index = 0
        self._previous_frame_start: float | None = None

    def update(self):
        """Updates the FPS calculation with the current frametime."""
        start, self._previous_frame_start = self._previous_frame_start, perf_counter()

        if start is None:
            return

        frametime = perf_counter() - start

        self._frame_times[self._frame_times_index] = frametime
        self._frame_times_index = (self._frame_times_index + 1) % self._fps_window

    @property
    def mean(self):
        """Returns the mean FPS over the last window_size frames."""
        return 1.0 / np.nanmean(self._frame_times)

    @property
    def stdev(self):
        """Returns the standard deviation of the FPS over the last window_size frames."""
        return np.nanstd(1.0 / self._frame_times)

    @property
    def stats(self):
        """Returns the mean and standard deviation of the FPS over the last window_size frames."""
        return {
            'fps': self.mean,
            'fps_std': self.stdev,
            'fps_last': 1.0 / self._frame_times[(self._frame_times_index - 1) % self._fps_window]
        }
