"""Components/ConfigSections/TrajectoryRenderer.py: Interactive trajectory definition and rendering section."""

import json
import math
import itertools
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from datetime import datetime
from enum import Enum
from itertools import count
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
from imgui_bundle import imgui, imgui_ctx, icons_fontawesome_6 as fa, portable_file_dialogs

from Cameras.Base import BaseCamera
from Datasets.utils import View
from ICGui.Components.AdvancedSeparator import advanced_separator
from ICGui.Components.ButtonBehavior import button_behavior
from ICGui.Components.Colors import UI_COLORS, apply_ui_color
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.Components.TensorInput import input_mat4
from ICGui.Components.Tooltip import tooltip
from ICGui.Controls import InputHandler
from ICGui.Controls.Camera import FlyingControls
from ICGui.Controls.Camera.utils import Animation
from ICGui.State.Volatile import CameraState, ColorMapState, GlobalState, OverlayState, TimeState, ViewerOutputState
from ICGui.util.Cameras import CAMERA_TYPES, DISTORTION_TYPES, camera_from_dict, camera_to_dict, get_camera_idx, \
    get_distortion_idx, interpolate_cameras, rescale_camera
from .CameraSection import CameraSection
from .Section import Section


class TrajectoryMode(Enum):
    SEGMENTED = 0
    TRAJECTORY = 1


class EasingFunction:
    """
    Collection of easing functions for interpolation between keyframes. Each
      function takes a parameter t in [0, 1] and returns a value in [0, 1]
      representing the eased progress between the keyframes.
    Some mathematical definitions adapted from https://easings.net/
    """
    @staticmethod
    def step_start(t: float) -> float:
        """Step function that jumps from 0 to 1 at t=0."""
        if t > 0:
            return 1
        return 0

    @staticmethod
    def step_end(t: float) -> float:
        """Step function that jumps from 0 to 1 at t=1."""
        if t < 1:
            return 0
        return 1

    @staticmethod
    def nearest(t: float) -> float:
        """Step function that jumps from 0 to 1 at t=0.5."""
        if t < 0.5:
            return 0
        return 1

    @staticmethod
    def linear(t: float):
        """Linear easing function."""
        return t

    @staticmethod
    def ease_in_cubic(t: float):
        """Cubic easing function that starts slow and accelerates."""
        return t * t * t

    @staticmethod
    def ease_out_cubic(t: float):
        """Cubic easing function that starts fast and decelerates."""
        return 1 - math.pow(1 - t, 3)

    @staticmethod
    def ease_in_out_cubic(t: float):
        """Cubic easing function that starts and ends slow, with a fast middle."""
        if t < 0.5:
            return 4 * t * t * t
        return 1 - math.pow(-2 * t + 2, 3) / 2

    @staticmethod
    def ease_in_back(t: float):
        """Back easing function that starts by moving slightly backwards, then fast forwards."""
        c1 = 1.70158
        c3 = c1 + 1
        return c3 * t * t * t - c1 * t * t

    @staticmethod
    def ease_out_back(t: float):
        """Back easing function that ends by overshooting the target, then settling back."""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * math.pow(t - 1, 3) + c1 * math.pow(t - 1, 2)

    @staticmethod
    def ease_in_out_back(t: float):
        """Back easing function that combines ease_in_back and ease_out_back."""
        c1 = 1.70158
        c2 = c1 * 1.525
        if t < 0.5:
            return (math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
        return (math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2

    @staticmethod
    def ease_out_elastic(t: float):
        """Elastic easing function that starts fast and decelerates with an elastic bounce."""
        c4 = (2 * math.pi) / 3

        if t <= 0:
            return 0
        if t >= 1:
            return 1
        return math.pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

    @staticmethod
    def ease_out_bounce(t: float):
        """Bounce easing function that starts fast and decelerates with a bounce."""
        n1 = 7.5625
        d1 = 2.75

        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            return n1 * (t - 1.5 / d1) * (t - 1.5 / d1) + 0.75
        elif t < 2.5 / d1:
            return n1 * (t - 2.25 / d1) * (t - 2.25 / d1) + 0.9375
        # else:
        return n1 * (t - 2.625 / d1) * (t - 2.625 / d1) + 0.984375


EASING_FUNCTION_MAP = {
    fun.replace('_', '-').title(): getattr(EasingFunction, fun)
    for fun in dir(EasingFunction)
    if not fun.startswith('_') and callable(getattr(EasingFunction, fun))
}
EASING_FUNCTION_NAMES = list(sorted(EASING_FUNCTION_MAP.keys()))  # Sorted list of easing function names for combo box


@dataclass
class Keyframe:
    keyframes: ClassVar[dict[int, 'Keyframe']] = {}

    c2w: np.ndarray
    camera: BaseCamera
    timestamp: float = 0.0
    duration: int = 30  # Duration in frames
    easing_function: str = 'Linear'

    identifier: int = field(init=False, default_factory=count().__next__)
    last_edited: datetime = field(init=False, default_factory=datetime.now)

    def __post_init__(self):
        self.keyframes[self.identifier] = self

    def __eq__(self, other):
        if not isinstance(other, Keyframe):
            return False
        return self.identifier == other.identifier

    def copy(self) -> 'Keyframe':
        """Returns a copy of this keyframe with a new identifier."""
        return Keyframe(c2w=self.c2w.copy(), camera=deepcopy(self.camera), timestamp=self.timestamp,
                        duration=self.duration, easing_function=self.easing_function)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the keyframe into a human-readable, JSON-compatible dict."""
        return {
            'c2w': self.c2w.reshape(-1).tolist(),
            'timestamp': float(self.timestamp),
            'duration': int(self.duration),
            'easing_function': self.easing_function,
            'camera': camera_to_dict(self.camera),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> 'Keyframe':
        """Deserializes a keyframe, falling back to the current GUI camera if no camera is given."""
        c2w = np.array(data['c2w'], dtype=np.float64).reshape((4, 4))
        camera = camera_from_dict(data['camera']) if 'camera' in data else deepcopy(CameraState().current_camera)
        return Keyframe(
            c2w=c2w,
            camera=camera,
            timestamp=float(data.get('timestamp', 0.0)),
            duration=int(data.get('duration', 30)),
            easing_function=str(data.get('easing_function', 'Linear')),
        )


@dataclass
class KeyframeSegmentedAnimation(Animation):
    """Class to handle keyframe animations for camera transformations."""
    keyframes: list[Keyframe]
    fps: int = 30
    loop: bool = False
    # Output resolution the keyframe cameras are adapted to, None for the current GUI resolution
    resolution: tuple[int, int] | None = None
    crop_to_resolution: bool = False
    maintain_pixel_aspect: bool = True
    camera_callback: Callable[[], None] | None = None  # Called after each update of the GUI camera

    _current_keyframe: int = field(init=False, default=0)
    _current_keyframe_start: int = field(init=False, default=0)
    _trajectory_controller: ClassVar[FlyingControls] = FlyingControls(np.eye(4))

    def run(self, obj: object, delta_time: float):
        """
        Runs the keyframe animation, updating the object's attributes and the GUI camera.

        Args:
            obj (object): The object whose attributes are being animated.
            delta_time (float): The time elapsed since the last update.
        """
        if self.finished:
            return
        self.t += delta_time

        current_frame = int(self.t * self.fps)
        while current_frame >= self._current_keyframe_start + self.keyframes[self._current_keyframe].duration:
            self._current_keyframe_start += self.keyframes[self._current_keyframe].duration
            self._current_keyframe += 1
            if not self.loop and self._current_keyframe > len(self.keyframes) - 2:
                # Since the last keyframe has no next keyframe to interpolate to, finish the animation when we reach it
                self.finish(obj)
                return
            elif self._current_keyframe > len(self.keyframes) - 1:
                # Wrap around to the beginning if looping, otherwise clamp to the end
                current_frame %= self._current_keyframe_start
                self.t %= (self._current_keyframe_start / self.fps)
                self._current_keyframe = 0
                self._current_keyframe_start = 0

        interp_t = (current_frame - self._current_keyframe_start) / self.keyframes[self._current_keyframe].duration
        from_keyframe = self.keyframes[self._current_keyframe]
        to_keyframe = self.keyframes[(self._current_keyframe + 1) % len(self.keyframes)]  # Wrap around to the first keyframe at the end
        c2w, timestamp, camera = self._interpolate(from_keyframe, to_keyframe, interp_t)

        setattr(obj, 'c2w', c2w)
        setattr(obj, '_timestamp', timestamp)
        self._set_camera(camera)

    def finish(self, obj: object):
        """Immediately finishes the animation, setting the object's attributes and the GUI camera to their final values."""
        self.t = 0.0
        self.finished = True
        if len(self.keyframes) == 0:
            return

        if self.loop:
            keyframe = self.keyframes[0]
        else:
            keyframe = self.keyframes[-1]
        self._trajectory_controller.c2w = keyframe.c2w
        setattr(obj, 'c2w', keyframe.c2w)
        setattr(obj, '_timestamp', keyframe.timestamp)
        self._set_camera(self._rescale(keyframe.camera))

    def get_all_frames(self, grouped: bool = False, c2w_only: bool = False) \
            -> list[View] | list[np.ndarray] | list[list[View]] | list[list[np.ndarray]]:
        keyframes = self.keyframes
        if len(keyframes) == 0:
            return []
        if self.loop:
            keyframes = keyframes + [keyframes[0]]  # Add first keyframe to the end for looping

        frames = []
        for from_keyframe, to_keyframe in itertools.pairwise(keyframes):
            if grouped:
                keyframe_frames = []
                frames.append(keyframe_frames)
            else:
                keyframe_frames = frames
            for frame in range(from_keyframe.duration):
                c2w, timestamp, camera = self._interpolate(from_keyframe, to_keyframe, frame / from_keyframe.duration,
                                                           with_camera=not c2w_only)
                keyframe_frames.append(c2w if c2w_only else View(
                    c2w=c2w,
                    camera=camera,
                    camera_index=-1,
                    timestamp=timestamp,
                ))

        # If not looping, add final keyframe as a single frame
        if not self.loop:
            if grouped:
                keyframe_frames = []
                frames.append(keyframe_frames)
            else:
                keyframe_frames = frames
            last_keyframe = self.keyframes[-1]
            if c2w_only:
                keyframe_frames.append(last_keyframe.c2w.copy())
            else:
                keyframe_frames.append(View(
                    c2w=last_keyframe.c2w,
                    camera=self._rescale(last_keyframe.camera),
                    camera_index=-1,
                    timestamp=last_keyframe.timestamp,
                ))

        return frames

    def _interpolate(self, from_keyframe: Keyframe, to_keyframe: Keyframe, interp_t: float,
                     with_camera: bool = True) -> tuple[np.ndarray, float, BaseCamera | None]:
        """Interpolates c2w, timestamp and (optionally) camera between two keyframes."""
        easing_function = EASING_FUNCTION_MAP.get(from_keyframe.easing_function, EasingFunction.linear)
        easing_t = easing_function(interp_t)

        self._trajectory_controller.c2w = from_keyframe.c2w
        from_position = self._trajectory_controller.position
        from_rotation = self._trajectory_controller.rotation
        self._trajectory_controller.c2w = to_keyframe.c2w
        to_position = self._trajectory_controller.position
        to_rotation = self._trajectory_controller.rotation

        setattr(self._trajectory_controller, 'position', self.lerp(from_position, to_position, easing_t))
        setattr(self._trajectory_controller, 'rotation', self.slerp(from_rotation, to_rotation, easing_t))
        timestamp = self.lerp(from_keyframe.timestamp, to_keyframe.timestamp, easing_t)
        camera = None
        if with_camera:
            camera = interpolate_cameras(self._rescale(from_keyframe.camera), self._rescale(to_keyframe.camera), easing_t)
        return self._trajectory_controller.c2w, timestamp, camera

    def _set_camera(self, camera: BaseCamera):
        CameraState().set_camera(camera)
        if self.camera_callback is not None:
            self.camera_callback()

    def _rescale(self, camera: BaseCamera) -> BaseCamera:
        """Adapts a keyframe camera to the output resolution."""
        width, height = self.resolution or (CameraState().width, CameraState().height)
        return rescale_camera(camera, width, height, self.crop_to_resolution, self.maintain_pixel_aspect)


@dataclass
class TrajectoryRendererSection(Section):
    camera_section: InitVar[CameraSection] = None

    name: str = f'{fa.ICON_FA_BEZIER_CURVE} Trajectory Renderer'
    always_open: bool = False
    default_open: bool = False
    change_highlight_duration: float = 1.0  # Seconds to highlight a keyframe after it's changed

    fps: int = 30  # Frames per second for the trajectory
    loop: bool = True  # Whether the trajectory should loop when played

    _camera_section: CameraSection = field(init=False)  # Renders keyframe camera settings, synced when the GUI camera is replaced
    _keyframes: list[Keyframe] = field(init=False, default_factory=list)
    _keyframe_selected: int = field(init=False, default=-1)  # Index of the currently selected keyframe for editing
    _keyframe_was_moved: bool = field(init=False, default=False)  # Keep track of frame movement to avoid (de)selection after dragging
    _matrices_as_text: bool = field(init=False, default=False)  # Whether to show matrices as text inputs instead of separate fields
    _show_time_as_seconds: bool = False  # Whether to show start/end/duration as seconds instead of frames

    _trajectory_mode: TrajectoryMode = field(init=False, default=TrajectoryMode.SEGMENTED)
    # Dummy controller to avoid modifying actual camera for temporary internal use, e.g. when editing / taking screenshots
    _trajectory_controller: FlyingControls = field(init=False, default_factory=lambda: FlyingControls(np.eye(4)))
    _waiting_for_frames: int = field(init=False, default=0)  # Number of frames we're waiting to be rendered after clicking "Render", used to show a loading indicator
    _show_progress_percent: bool = field(init=False, default=False)  # Whether to show progress as percentage or frame count when rendering trajectory
    _trajectory_modified: bool = False  # Whether to rebuild the trajectory visualization next frame due to keyframe / trajectory modifications

    _resolution_override: tuple[int, int] | None = field(init=False, default=None)
    _crop_to_resolution: bool = field(init=False, default=False)  # Whether to crop the view to the overridden resolution when taking screenshots
    _maintain_pixel_aspect: bool = field(init=False, default=True)  # Whether the pixel aspect ratio is allowed to change from overridden resolution

    _TRAJECTORY_MODE_NAMES: ClassVar[list[str]] = [mode.name.replace('_', ' ').title().replace('Trajectory', 'Trajectory (NYI)') for mode in TrajectoryMode]
    _FILE_DIALOG_TIMEOUT: ClassVar[int] = 1000 // 30  # ms, time to wait each frame for file dialogs

    _confirm_clear_open: bool = field(init=False, default=False)  # Whether the confirmation dialog modal is open
    _show_subset_menu: bool = field(init=False, default=False)  # Whether to show subset context menu
    _last_error_message: str = field(init=False, default='')
    _error_popup_open: bool = field(init=False, default=False)
    _export_file_dialog: 'portable_file_dialogs.save_file | None' = field(init=False, default=None)
    _import_file_dialog: 'portable_file_dialogs.open_file | None' = field(init=False, default=None)

    def __post_init__(self, camera_section: CameraSection):
        super().__post_init__()
        self._camera_section = camera_section
        OverlayState().additional_overlay_data['Trajectory Frustums'] = {
            'keyframe_poses': [],
            'interpolated_poses': [],
            'selected_keyframe_poses': [],
            'does_loop': self.loop,
            '_vis_version': 0,  # Incremental counter to trigger overlay updates only on changes to trajectory data
        }

    def update(self):
        if not self._trajectory_modified:
            return
        self._trajectory_modified = False

        # Rebuild trajectory visualization
        animation = KeyframeSegmentedAnimation(keyframes=[df for df in self._keyframes], fps=self.fps, loop=self.loop)
        vis_data = OverlayState().additional_overlay_data['Trajectory Frustums']
        vis_data['keyframe_poses'] = [kf.c2w for kf in self._keyframes if kf.identifier != self._keyframe_selected]
        vis_data['selected_keyframe_poses'] = [kf.c2w for kf in self._keyframes if kf.identifier == self._keyframe_selected]
        vis_data['interpolated_poses'] = animation.get_all_frames(grouped=True, c2w_only=True)
        vis_data['does_loop'] = self.loop
        vis_data['_vis_version'] += 1  # Increment visualization version to trigger overlay update

    @property
    def _render_resolution(self) -> tuple[int, int]:
        """Resolution of rendered trajectories and keyframe screenshots.
        Returns the override if set, the first keyframe's if present, the current window size otherwise."""
        if self._resolution_override:
            return self._resolution_override
        if len(self._keyframes) > 0:
            first_camera = self._keyframes[0].camera
            return (first_camera.width, first_camera.height)
        return GlobalState().window_size

    def _render(self):
        is_disabled = False
        if self._waiting_for_frames > 0 and self._waiting_for_frames > GlobalState().shared.trajectory_progress_num_frames:
            self._render_progress()
            imgui.separator()
            is_disabled = True
            imgui.begin_disabled()
        else:
            self._waiting_for_frames = 0

        self._render_settings()
        imgui.spacing()
        imgui.separator_text(f'{fa.ICON_FA_LAYER_GROUP} Keyframes')
        self._render_keyframe_list()

        imgui.spacing()
        self._render_keyframe_editor()

        if is_disabled:
            imgui.end_disabled()

    def _render_progress(self):
        imgui.text(f'{fa.ICON_FA_HOURGLASS} Rendering trajectory with {self._waiting_for_frames} frames...')

        progress = GlobalState().shared.trajectory_progress_num_frames / self._waiting_for_frames
        if self._show_progress_percent:
            overlay_text = f'{progress * 100:.1f}%'
        else:
            overlay_text = f'{GlobalState().shared.trajectory_progress_num_frames} / {self._waiting_for_frames}'

        with button_behavior('Rendering Progress') as button:
            imgui.progress_bar(progress, overlay=overlay_text, size_arg=(0, 0))
            imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
            imgui.text('Rendering...')
        self._show_progress_percent ^= button.pressed

        with apply_ui_color('ERROR'):
            if imgui.button(f'{fa.ICON_FA_XMARK} Cancel'):
                GlobalState().shared.cancel_trajectory_rendering()
                self._waiting_for_frames = 0

    def _render_settings(self):
        changed, new_mode = imgui.combo('Trajectory Mode', self._trajectory_mode.value,
                                        self._TRAJECTORY_MODE_NAMES)
        if changed:
            self._trajectory_mode = TrajectoryMode(new_mode)

        changed, fps = imgui.input_int('Frames Per Second', self.fps)
        if changed:
            self.fps = max(1, fps)  # Clamp to at least 1 FPS
        _, self._show_time_as_seconds = styled_toggle('Show Time as Seconds', self._show_time_as_seconds)
        help_indicator('Show start/end/duration as seconds instead of frames.')

        changed, self.loop = styled_toggle('Loop Trajectory', self.loop)
        help_indicator('Whether the trajectory should loop back to the beginning. This adds a duration and easing '
                       'function to the last frame to interpolate back to the first frame.')
        if changed:
            self._trajectory_modified = True

        _, override_resolution = styled_toggle(
            'Override Render Resolution',
            self._resolution_override is not None
        )
        help_indicator('Resolution of rendered trajectories and keyframe screenshots. If not overridden, the '
                       'resolution of the first keyframe camera is used.')
        if override_resolution:
            # On first enabling, set override resolution to the current render resolution
            if self._resolution_override is None:
                self._resolution_override = self._render_resolution

            _, resolution = imgui.input_int2('Render Resolution', list(self._resolution_override))  # type: ignore
            self._resolution_override = (max(1, resolution[0]), max(1, resolution[1]))  # Clamp to > 0

            _, self._crop_to_resolution = styled_toggle('Crop to Resolution', self._crop_to_resolution)
            help_indicator('Crop rendered view to the target resolution instead of stretching. '
                           'This results in an effective change in the field of view.')
            if not self._crop_to_resolution:
                _, self._maintain_pixel_aspect = styled_toggle('Maintain Pixel Aspect Ratio', self._maintain_pixel_aspect)
                help_indicator('When stretching, maintain the current pixel aspect ratio, i.e. maintain the field '
                               'of view in one axis and grow the field of view in the other axis if necessary.')
        else:
            self._resolution_override = None

        # Draw action buttons
        disabled = False
        if len(self._keyframes) < 1:
            imgui.begin_disabled()
            disabled = True

        control_scheme = GlobalState().input_manager.control_scheme
        animation_running = control_scheme.animation_running
        btn_text = f'{fa.ICON_FA_STOP} Stop Preview' if animation_running else f'{fa.ICON_FA_CIRCLE_PLAY} Preview'
        if imgui.button(btn_text):
            if animation_running:
                control_scheme.stop_animation()
            else:
                control_scheme.set_animation(KeyframeSegmentedAnimation(
                    keyframes=self._keyframes,
                    fps=self.fps,
                    loop=self.loop,
                    camera_callback=self._camera_section.sync_with_current_camera,
                ))
        tooltip('Stop the current preview animation.' if animation_running else 'Run a preview of this animation in the GUI at the selected FPS.')

        btn_text = f'{fa.ICON_FA_CLAPPERBOARD} Render'
        space_required = imgui.calc_text_size(btn_text).x + 2 * imgui.get_style().frame_padding.x
        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        if space_required > imgui.get_content_region_avail().x:
            imgui.new_line()  # Move to next line if not enough space for button
        if imgui.button(btn_text):
            animation = KeyframeSegmentedAnimation(
                keyframes=self._keyframes,
                fps=self.fps,
                loop=self.loop,
                resolution=self._render_resolution,
                crop_to_resolution=self._crop_to_resolution,
                maintain_pixel_aspect=self._maintain_pixel_aspect,
            )
            frames = animation.get_all_frames()
            self._waiting_for_frames = len(frames)
            GlobalState().shared.render_trajectory(
                frames,
                ViewerOutputState().selected_output,
                ColorMapState().dict,
            )
        tooltip('Exports this animation as a set of frames to the output directory. This may take a while.')

        if disabled:
            imgui.end_disabled()

    def _render_keyframe_list(self):
        input_manager: InputHandler = GlobalState().input_manager
        swap_frames = (-1, -1)  # (index1, index2) of keyframes to swap for reordering
        delete_frame = None  # Keyframe index to delete after the loop, max 1 per frame
        clone_frame = None  # Keyframe index to clone after the loop, max 1 per frame

        table_height = (
            imgui.get_frame_height()
            + min(10, len(self._keyframes))  # Limit to 10 rows before scrolling
            * (imgui.get_frame_height() + imgui.get_style().cell_padding.y * 2)
        )
        with imgui_ctx.begin_table(
            'Keyframe Table', 5,
            imgui.TableFlags_.borders_inner | imgui.TableFlags_.resizable | imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_borders_in_body_until_resize | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.no_saved_settings,
            outer_size=(0, table_height)
        ) as table:
            if not table.visible:
                return
            imgui.table_setup_column('ID', imgui.TableColumnFlags_.width_stretch, 50)
            if self._show_time_as_seconds:
                imgui.table_setup_column('Start Time (s)', imgui.TableColumnFlags_.width_stretch, 100)
                imgui.table_setup_column('End Time (s)', imgui.TableColumnFlags_.width_stretch, 100)
            else:
                imgui.table_setup_column('Start Frame', imgui.TableColumnFlags_.width_stretch, 100)
                imgui.table_setup_column('End Frame', imgui.TableColumnFlags_.width_stretch, 100)
            imgui.table_setup_column('Duration', imgui.TableColumnFlags_.width_stretch, 100)
            imgui.table_setup_column('Actions', imgui.TableColumnFlags_.width_stretch, 100)

            imgui.table_headers_row()

            current_frame = 0
            for i, keyframe in enumerate(self._keyframes):
                imgui.push_id(keyframe.identifier)
                imgui.push_style_color(imgui.Col_.frame_bg, 0)
                imgui.push_style_var(imgui.StyleVar_.frame_rounding, 4)
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                col_height = imgui.get_frame_height()

                # Highlight entire row if currently editing
                if self._keyframe_selected == keyframe.identifier:
                    imgui.table_set_bg_color(
                        imgui.TableBgTarget_.row_bg0,
                        imgui.color_convert_float4_to_u32(imgui.get_style_color_vec4(imgui.Col_.frame_bg_active)),
                    )
                if (datetime.now() - keyframe.last_edited).total_seconds() < self.change_highlight_duration:
                    opacity = 1 - (datetime.now() - keyframe.last_edited).total_seconds() / self.change_highlight_duration  # Fade out over time
                    base_color = UI_COLORS['WARNING'][imgui.Col_.frame_bg]
                    imgui.table_set_bg_color(
                        imgui.TableBgTarget_.row_bg1,
                        imgui.color_convert_float4_to_u32((base_color.x, base_color.y, base_color.z, opacity)),
                    )

                # Save start position for drag handle
                text_start = imgui.get_cursor_pos_x()
                # Save position for index text (centered in the column, accounting for the drag handle icon)
                text_pos_x = (
                    imgui.get_cursor_pos_x() + imgui.calc_text_size(f'{fa.ICON_FA_GRIP_LINES}').x + imgui.get_style().item_inner_spacing.x
                    + max(0, imgui.get_column_width(0) - imgui.calc_text_size(f'{fa.ICON_FA_GRIP_LINES}').x - imgui.get_style().item_inner_spacing.x - imgui.calc_text_size(f'{keyframe.identifier}').x) / 2
                )
                text_pos_y = imgui.get_cursor_pos_y() + col_height / 2 - imgui.calc_text_size(f'{i}').y / 2

                # Place selectable for editing & dragging
                imgui.push_style_color(imgui.Col_.header, 0)  # Don't show selection highlight on the selectable itself, since we're highlighting the whole row
                if imgui.selectable(f'##Selectable_{keyframe.identifier}', keyframe.identifier == self._keyframe_selected, size=(0, col_height))[0]:
                    if not self._keyframe_was_moved:  # Only toggle selection if we didn't just move the frame
                        if keyframe.identifier == self._keyframe_selected:
                            self._keyframe_selected = -1  # Deselect if already selected
                        else:
                            self._keyframe_selected = keyframe.identifier
                        self._trajectory_modified = True
                if imgui.is_item_activated():
                    self._keyframe_was_moved = False # Reset move flag on new selection
                imgui.pop_style_color()

                # Handle drag & drop for reordering keyframes
                if imgui.is_item_active() and not imgui.is_item_hovered():
                    n_next = i + (-1 if imgui.get_mouse_drag_delta(0).y < 0 else 1)
                    if 0 <= n_next < len(self._keyframes):
                        swap_frames = (i, n_next)
                        imgui.reset_mouse_drag_delta()

                # Draw drag handle and index text
                imgui.set_cursor_pos_x(text_start)
                imgui.set_cursor_pos_y(text_pos_y)
                imgui.text(f'{fa.ICON_FA_GRIP_LINES}')
                imgui.set_cursor_pos_x(text_pos_x)
                imgui.set_cursor_pos_y(text_pos_y)
                imgui.text(f'{keyframe.identifier}')

                # Draw start frame input
                imgui.table_set_column_index(1)
                imgui.begin_disabled(i == 0)
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_style().item_inner_spacing.x)
                imgui.set_next_item_width(imgui.get_column_width(1) - imgui.get_style().item_inner_spacing.x * 2)
                if self._show_time_as_seconds:
                    changed, val_seconds = imgui.input_float(f'##Start Time (s)', current_frame / self.fps, format='%.3f')
                    val = int(val_seconds * self.fps + 0.5)  # Round to nearest frame
                else:
                    changed, val = imgui.input_int(f'##Start Frame', current_frame)
                if changed and i > 0:
                    # Update previous frame duration
                    duration_before = self._keyframes[i - 1].duration
                    self._keyframes[i - 1].duration = max(1, self._keyframes[i - 1].duration + (val - current_frame))
                    current_frame += self._keyframes[i - 1].duration - duration_before  # Update current_frame
                    self._trajectory_modified = True
                imgui.end_disabled()

                # When not looping, disable end frame / duration inputs for the last frame
                if not self.loop and i == len(self._keyframes) - 1:
                    imgui.begin_disabled()  # Disable end frame input for last frame if not looping, since it's determined by the duration and can't be changed independently

                # Draw end frame input
                imgui.table_set_column_index(2)
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_style().item_inner_spacing.x)
                imgui.set_next_item_width(imgui.get_column_width(2) - imgui.get_style().item_inner_spacing.x * 2)
                if self._show_time_as_seconds:
                    changed, val_seconds = imgui.input_float(
                        f'##End Time (s)',
                        (current_frame + keyframe.duration) / self.fps,
                        format='%.3f'
                    )
                    val = int(val_seconds * self.fps + 0.5)  # Round to nearest frame
                else:
                    changed, val = imgui.input_int(f'##End Frame', current_frame + keyframe.duration)
                if changed:
                    keyframe.duration = max(1, val - current_frame)
                    self._trajectory_modified = True

                # Draw duration input
                imgui.table_set_column_index(3)
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + imgui.get_style().item_inner_spacing.x)
                imgui.set_next_item_width(imgui.get_column_width(3) - imgui.get_style().item_inner_spacing.x * 2)
                if self._show_time_as_seconds:
                    changed, val_seconds = imgui.input_float(f'##Duration (s)', keyframe.duration / self.fps, format='%.3f')
                    val = int(val_seconds * self.fps + 0.5)  # Round to nearest frame
                else:
                    changed, val = imgui.input_int(f'##Duration', keyframe.duration)
                if changed:
                    keyframe.duration = max(1, val)
                    self._trajectory_modified = True
                if not self.loop and i == len(self._keyframes) - 1:
                    imgui.end_disabled()

                current_frame += keyframe.duration

                imgui.table_set_column_index(4)
                delete_requested, clone_requested = self._draw_action_buttons(keyframe.identifier)
                delete_frame = delete_requested if delete_requested is not None else delete_frame
                clone_frame = clone_requested if clone_requested is not None else clone_frame

                imgui.pop_id()
                imgui.pop_style_var()
                imgui.pop_style_color()

        # Perform swap / deletions after the loop to avoid modifying the list while iterating
        if swap_frames != (-1, -1):
            self._keyframes[swap_frames[0]], self._keyframes[swap_frames[1]] = (
                self._keyframes[swap_frames[1]], self._keyframes[swap_frames[0]])
            self._keyframe_was_moved = True
            self._trajectory_modified = True
        if delete_frame is not None:
            self._keyframes = [kf for kf in self._keyframes if kf.identifier != delete_frame]
            Keyframe.keyframes.pop(delete_frame, None)
            if clone_frame is not None and clone_frame == delete_frame:
                clone_frame = None  # Don't clone if we're deleting the same frame
            self._trajectory_modified = True
        if clone_frame is not None:
            kf_to_clone = Keyframe.keyframes[clone_frame]
            new_kf = kf_to_clone.copy()
            self._keyframes.insert(self._keyframes.index(kf_to_clone) + 1, new_kf)  # Insert cloned frame right after the original
            self._keyframe_selected = new_kf.identifier  # Select the newly cloned keyframe for editing
            self._trajectory_modified = True

        ## Action Buttons
        button_spacing = imgui.get_style().item_inner_spacing.x
        frame_padding = imgui.get_style().frame_padding.x

        # Handle adding new keyframe
        if imgui.button(f'{fa.ICON_FA_PLUS} Add Keyframe'):
            duration = 30
            easing_function = 'Linear'
            if len(self._keyframes) > 0:
                duration = self._keyframes[-1].duration  # Default to last keyframe's duration
                easing_function = self._keyframes[-1].easing_function  # Default to last keyframe's easing function
            if not self.loop and len(self._keyframes) > 1:
                # If not looping, set the default duration to the second to last frame instead, since the last
                # frame's duration is not editable
                duration = self._keyframes[-2].duration

            self._keyframes.append(Keyframe(
                c2w=input_manager.control_scheme.c2w,
                camera=deepcopy(CameraState().current_camera),
                timestamp=TimeState().timestamp,
                duration=duration,
                easing_function=easing_function,
            ))
            self._keyframe_selected = self._keyframes[-1].identifier  # Select the newly added keyframe for editing
            self._trajectory_modified = True

        # Add all from subset with context menu
        btn_text = f'{fa.ICON_FA_LIST} Add All From Subset'
        space_required = imgui.calc_text_size(btn_text).x + 2 * frame_padding
        imgui.same_line(spacing=button_spacing)
        if space_required > imgui.get_content_region_avail().x:
            imgui.new_line()
        if imgui.button(btn_text):
            # Show context menu
            imgui.open_popup('##subset_menu')
        tooltip('Select a subset to add all poses as keyframes')

        # Context menu for subsets
        if imgui.begin_popup('##subset_menu'):
            camera_state = CameraState()
            available_subsets = list(camera_state.dataset_poses.keys()) if camera_state.dataset_poses else []
            
            if len(available_subsets) == 0:
                imgui.text('No subsets available')
            else:
                imgui.separator_text('Add poses from subset:')
                for subset in available_subsets:
                    subset_display = subset.replace('_', ' ').title()
                    if imgui.menu_item(subset_display, '', False)[0]:
                        # Add all poses from this subset as keyframes
                        poses = camera_state.dataset_poses.get(subset, [])
                        if len(poses) > 0:
                            # Default duration/easing from last keyframe if present
                            default_duration = 30
                            default_easing = 'Linear'
                            if len(self._keyframes) > 0:
                                default_duration = self._keyframes[-1].duration
                                default_easing = self._keyframes[-1].easing_function

                            for p in poses:
                                self._keyframes.append(Keyframe(
                                    c2w=p.c2w_numpy,
                                    camera=deepcopy(camera_state.current_camera),
                                    timestamp=p.timestamp,
                                    duration=default_duration,
                                    easing_function=default_easing,
                                ))
                            if len(self._keyframes) > 0:
                                self._keyframe_selected = self._keyframes[-1].identifier
                        self._trajectory_modified = True
            imgui.end_popup()

        btn_text = f'{fa.ICON_FA_FILE_EXPORT} Export Keyframes'
        space_required = imgui.calc_text_size(btn_text).x + 2 * frame_padding
        imgui.same_line(spacing=button_spacing)
        if space_required > imgui.get_content_region_avail().x:
            imgui.new_line()
        if imgui.button(btn_text):
            # Open save dialog
            if len(self._keyframes) == 0:
                self._last_error_message = 'No keyframes to export.'
                self._error_popup_open = True
            else:
                self._export_file_dialog = portable_file_dialogs.save_file(
                    title='Export Keyframes as JSON',
                    default_path=str(Path.home().absolute()),
                    filters=['JSON files (*.json)|*.json', 'All files (*.*)|*.*'],
                )
        tooltip('Export keyframes to a JSON file')

        btn_text = f'{fa.ICON_FA_FILE_IMPORT} Import Keyframes'
        space_required = imgui.calc_text_size(btn_text).x + 2 * frame_padding
        imgui.same_line(spacing=button_spacing)
        if space_required > imgui.get_content_region_avail().x:
            imgui.new_line()
        if imgui.button(btn_text):
            # Open file dialog
            self._import_file_dialog = portable_file_dialogs.open_file(
                title='Import Keyframes from JSON',
                default_path=str(Path.home().absolute()),
                filters=['JSON files (*.json)|*.json', 'All files (*.*)|*.*'],
            )
        tooltip('Import keyframes from a JSON file')

        btn_text = f'{fa.ICON_FA_COPY} Copy Keyframes'
        space_required = imgui.calc_text_size(btn_text).x + 2 * frame_padding
        imgui.same_line(spacing=button_spacing)
        if space_required > imgui.get_content_region_avail().x:
            imgui.new_line()
        if imgui.button(btn_text):
            # Build JSON and copy to clipboard
            if len(self._keyframes) == 0:
                self._last_error_message = 'No keyframes to copy.'
                self._error_popup_open = True
            else:
                json_str = json.dumps([kf.to_dict() for kf in self._keyframes], indent=2)
                try:
                    imgui.set_clipboard_text(json_str)
                except Exception as e:
                    self._last_error_message = f'Failed to copy to clipboard: {e}'
                    self._error_popup_open = True
        tooltip('Copy keyframes as JSON to clipboard')

        btn_text = f'{fa.ICON_FA_PASTE} Paste Keyframes'
        space_required = imgui.calc_text_size(btn_text).x + 2 * frame_padding
        imgui.same_line(spacing=button_spacing)
        if space_required > imgui.get_content_region_avail().x:
            imgui.new_line()
        if imgui.button(btn_text):
            # Read from clipboard and import
            try:
                clipboard_text = imgui.get_clipboard_text()
                if not clipboard_text:
                    self._last_error_message = 'Clipboard is empty.'
                    self._error_popup_open = True
                else:
                    try:
                        parsed = json.loads(clipboard_text)
                    except Exception as e:
                        self._last_error_message = f'Clipboard does not contain valid JSON: {e}'
                        self._error_popup_open = True
                        parsed = None

                    if isinstance(parsed, list):
                        self._add_keyframes_from_json(parsed)
                    elif parsed is not None:
                        self._last_error_message = 'Clipboard JSON is not a list of keyframes.'
                        self._error_popup_open = True
            except Exception as e:
                self._last_error_message = f'Failed to read from clipboard: {e}'
                self._error_popup_open = True
            self._trajectory_modified = True
        tooltip('Paste keyframes from clipboard (JSON format)')

        btn_text = f'{fa.ICON_FA_TRASH} Clear Keyframes'
        space_required = imgui.calc_text_size(btn_text).x + 2 * frame_padding
        imgui.same_line(spacing=button_spacing)
        if space_required > imgui.get_content_region_avail().x:
            imgui.new_line()
        with apply_ui_color('ERROR'):
            if imgui.button(btn_text):
                # Open confirmation popup
                self._confirm_clear_open = True
        tooltip('Clear all keyframes (confirmation required)')

        # Confirmation modal for clearing keyframes
        if self._confirm_clear_open:
            imgui.open_popup('Confirm Clear Keyframes')
            self._confirm_clear_open = False
        if imgui.begin_popup_modal('Confirm Clear Keyframes', flags=imgui.WindowFlags_.always_auto_resize)[0]:
            imgui.text('Are you sure you want to clear ALL keyframes? This action cannot be undone.')
            imgui.spacing()
            imgui.spacing()
            if imgui.button('Yes, Clear'):
                # Remove from global registry as well
                for kf in list(self._keyframes):
                    Keyframe.keyframes.pop(kf.identifier, None)
                self._keyframes = []
                self._keyframe_selected = -1
                self._trajectory_modified = True
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()

        # Handle file dialogs
        self._handle_export_file_dialog()
        self._handle_import_file_dialog()

        # Error popup handling
        if self._error_popup_open:
            imgui.open_popup('Keyframes Error')
            self._error_popup_open = False
        if imgui.begin_popup_modal('Keyframes Error', flags=imgui.WindowFlags_.always_auto_resize)[0]:
            imgui.text(self._last_error_message)
            imgui.separator()
            if imgui.button('OK'):
                imgui.close_current_popup()
            imgui.end_popup()

    def _render_keyframe_editor(self):
        if self._keyframe_selected is None or self._keyframe_selected == -1 or self._keyframe_selected not in Keyframe.keyframes:
            return  # No keyframe selected

        if advanced_separator(f'{fa.ICON_FA_PEN} Edit Keyframe {self._keyframe_selected}', f'{fa.ICON_FA_XMARK}'):
            self._keyframe_selected = -1  # Deselect keyframe to close editor
            return

        delete_frame, clone_frame = self._draw_action_buttons(self._keyframe_selected, verbose=True)  # Draw action buttons in editor as well for convenience
        if delete_frame is not None:
            self._keyframes = [kf for kf in self._keyframes if kf.identifier != delete_frame]
            Keyframe.keyframes.pop(delete_frame, None)
            self._trajectory_modified = True
            return
        if clone_frame is not None:
            kf_to_clone = Keyframe.keyframes[clone_frame]
            new_kf = kf_to_clone.copy()
            self._keyframes.insert(self._keyframes.index(kf_to_clone) + 1, new_kf)  # Insert cloned frame right after the original
            self._keyframe_selected = new_kf.identifier  # Select the newly cloned keyframe for editing
            self._trajectory_modified = True

        keyframe = Keyframe.keyframes[self._keyframe_selected]

        imgui.spacing()

        # Duration editor
        cursor_x = imgui.get_cursor_pos_x()
        cursor_y = imgui.get_cursor_pos_y()
        dur_icon = f'{fa.ICON_FA_HOURGLASS}'
        icon_size = imgui.calc_text_size(dur_icon)
        imgui.set_cursor_pos_y(cursor_y + imgui.get_frame_height() + imgui.get_style().item_spacing.y / 2.0 - icon_size.y / 2.0)
        imgui.text(dur_icon)
        imgui.set_cursor_pos_x(cursor_x + icon_size.x + imgui.get_style().item_spacing.x)
        imgui.set_cursor_pos_y(cursor_y)
        cursor_x_after_icon = imgui.get_cursor_pos_x()

        if self._show_time_as_seconds:
            changed, duration_seconds = imgui.input_float('Duration (s)##kf_editor', keyframe.duration / max(1, self.fps), format='%.3f')
            if changed:
                keyframe.duration = max(1, int(duration_seconds * self.fps + 0.5))
                keyframe.last_edited = datetime.now()
                self._trajectory_modified = True
        else:
            changed, duration = imgui.input_int('Duration##kf_editor', keyframe.duration)
            if changed:
                keyframe.duration = max(1, duration)
                keyframe.last_edited = datetime.now()
                self._trajectory_modified = True

        imgui.set_cursor_pos_x(cursor_x_after_icon)
        if imgui.button('Set All Keyframes to This Duration'):
            for idx, kf in enumerate(self._keyframes):
                kf.duration = keyframe.duration
                kf.last_edited = datetime.now()
            self._trajectory_modified = True

        imgui.spacing()

        # Easing function editor
        cursor_x = imgui.get_cursor_pos_x()
        cursor_y = imgui.get_cursor_pos_y()
        easing_icon = f'{fa.ICON_FA_CLOCK}'
        icon_size = imgui.calc_text_size(easing_icon)
        imgui.set_cursor_pos_y(cursor_y + imgui.get_frame_height() + imgui.get_style().item_spacing.y / 2.0 - icon_size.y / 2.0)
        imgui.text(easing_icon)
        imgui.set_cursor_pos_x(cursor_x + icon_size.x + imgui.get_style().item_spacing.x)
        imgui.set_cursor_pos_y(cursor_y)
        cursor_x = imgui.get_cursor_pos_x()
        changed, easing_index = imgui.combo(
            'Easing Function',
            EASING_FUNCTION_NAMES.index(keyframe.easing_function),
            EASING_FUNCTION_NAMES
        )
        if changed:
            keyframe.easing_function = EASING_FUNCTION_NAMES[easing_index]
            keyframe.last_edited = datetime.now()  # Mark keyframe as edited to show highlight
            self._trajectory_modified = True
        imgui.set_cursor_pos_x(cursor_x)
        if imgui.button('Update All Keyframes to This Easing Function'):
            for kf in self._keyframes:
                kf.easing_function = keyframe.easing_function
                kf.last_edited = datetime.now()  # Mark all keyframes as edited to show highlight
            self._trajectory_modified = True

        imgui.spacing()
        imgui.spacing()

        changed, c2w = input_mat4(f'{fa.ICON_FA_LOCATION_CROSSHAIRS} Camera to World Transformation', keyframe.c2w,
                                  as_separate=not self._matrices_as_text)
        _, self._matrices_as_text = styled_toggle('Show Matrix as Text', self._matrices_as_text)
        help_indicator('Show matrix as a multi-line text input, to support copy & paste. Changes are applied '
                       'on enter/focus loss of the text field. Valid formats include any set of 16 numbers, '
                       'separated by whitespace, commas or square brackets.')
        if changed:
            # Use dummy controller to sanitize the input, e.g. by enforcing orthogonality
            self._trajectory_controller.c2w = c2w
            keyframe.c2w = self._trajectory_controller.c2w
            keyframe.last_edited = datetime.now()  # Mark keyframe as edited to show highlight
            self._trajectory_modified = True

        if not TimeState().disabled:
            changed, keyframe.timestamp = imgui.slider_float('Timestamp', keyframe.timestamp, 0.0, 1.0, format='%.3f')
            if changed:
                keyframe.last_edited = datetime.now()  # Mark keyframe as edited to show highlight
                self._trajectory_modified = True

        imgui.spacing()
        self._render_keyframe_camera(keyframe)

    def _render_keyframe_camera(self, keyframe: Keyframe):
        imgui.text(f'{fa.ICON_FA_VIDEO} Camera')
        help_indicator('Camera settings are interpolated between keyframes. Changes of the camera model or '
                       'distortion type cut hard at the next keyframe. Cameras are rescaled to the render resolution.')
        imgui.push_id('keyframe_camera')  # Avoid ID clashes with the camera section
        camera_idx = get_camera_idx(keyframe.camera)
        changed_model, new_camera_idx = imgui.combo('Camera Model', camera_idx, [cam.name for cam in CAMERA_TYPES])
        changed = changed_model and new_camera_idx != camera_idx
        if changed:
            keyframe.camera = CameraSection.convert_camera(keyframe.camera, new_camera_idx)
        changed_resolution, resolution = imgui.input_int2('Resolution', [keyframe.camera.width, keyframe.camera.height])  # type: ignore
        if changed_resolution:
            keyframe.camera.width, keyframe.camera.height = max(1, resolution[0]), max(1, resolution[1])  # Clamp to > 0
        help_indicator('Reference resolution of the focal length and projection center (in px) of this keyframe. '
                       'Does not affect the GUI camera or window.')
        changed |= changed_resolution
        changed |= self._camera_section.render_camera_settings(keyframe.camera)
        if hasattr(keyframe.camera, 'distortion'):
            distortion_idx = get_distortion_idx(keyframe.camera.distortion)
            changed_distortion, new_distortion_idx = imgui.combo('Distortion Type', distortion_idx + 1,
                                                                 ['None', *[dist.name for dist in DISTORTION_TYPES]])
            new_distortion_idx -= 1  # Subtract 1, because None is not a real distortion type
            if changed_distortion and new_distortion_idx != distortion_idx:
                keyframe.camera.distortion = None if new_distortion_idx < 0 else CameraSection.create_distortion(new_distortion_idx)
                changed = True
            changed |= self._camera_section.render_distortion_settings(keyframe.camera)
        imgui.pop_id()
        if changed:
            keyframe.last_edited = datetime.now()  # Mark keyframe as edited to show highlight

        if imgui.button('Set Camera to Current'):
            keyframe.camera = deepcopy(CameraState().current_camera)
            keyframe.last_edited = datetime.now()
        tooltip("Set this keyframe's camera to the current GUI camera, keeping its view")

        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        if imgui.button('Set All Keyframes to This Camera'):
            for kf in self._keyframes:
                kf.camera = deepcopy(keyframe.camera)
                kf.last_edited = datetime.now()

    def _draw_action_buttons(self, keyframe_identifier: int, verbose: bool = False) -> tuple[int | None, int | None]:
        keyframe = Keyframe.keyframes[keyframe_identifier]
        state = GlobalState()
        input_manager: InputHandler = state.input_manager
        delete_frame = None  # Keyframe index to delete after the loop, max 1 per frame
        clone_frame = None  # Keyframe index to clone after the loop, max 1 per frame

        # Draw action buttons
        btn_text = f'{fa.ICON_FA_TRASH} Delete Keyframe' if verbose else f'{fa.ICON_FA_TRASH}'
        if imgui.button(btn_text):
            delete_frame = keyframe_identifier
        tooltip('Delete this keyframe')

        btn_text = f'{fa.ICON_FA_CLONE} Duplicate Keyframe' if verbose else f'{fa.ICON_FA_CLONE}'
        space_required = imgui.calc_text_size(btn_text).x + 2 * imgui.get_style().frame_padding.x
        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        if verbose and space_required > imgui.get_content_region_avail().x:
            imgui.new_line()  # Move to next line if not enough space for button and tooltip
        if imgui.button(btn_text):
            clone_frame = keyframe_identifier
        tooltip('Duplicate this keyframe')

        btn_text = f'{fa.ICON_FA_ARROWS_TO_EYE} Teleport to Keyframe' if verbose else f'{fa.ICON_FA_ARROWS_TO_EYE}'
        space_required = imgui.calc_text_size(btn_text).x + 2 * imgui.get_style().frame_padding.x
        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        if verbose and space_required > imgui.get_content_region_avail().x:
            imgui.new_line()  # Move to next line if not enough space for button and tooltip
        if imgui.button(btn_text):
            input_manager.control_scheme.ease_to(to_c2w=keyframe.c2w)
            CameraState().set_camera(rescale_camera(keyframe.camera, CameraState().width, CameraState().height))
            self._camera_section.sync_with_current_camera()
        tooltip("Teleport camera to this keyframe's view and apply its camera")

        btn_text = f'{fa.ICON_FA_EYE_DROPPER} Reset Keyframe View to Current' if verbose else f'{fa.ICON_FA_EYE_DROPPER}'
        space_required = imgui.calc_text_size(btn_text).x + 2 * imgui.get_style().frame_padding.x
        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        if verbose and space_required > imgui.get_content_region_avail().x:
            imgui.new_line()  # Move to next line if not enough space for button and tooltip
        if imgui.button(btn_text):
            keyframe.c2w = input_manager.control_scheme.c2w
            keyframe.camera = deepcopy(CameraState().current_camera)
            keyframe.last_edited = datetime.now()  # Mark keyframe as edited to show highlight
            self._trajectory_modified = True
        tooltip("Set this keyframe's view and camera to the current camera view")

        btn_text = f'{fa.ICON_FA_CAMERA} Take Screenshot at Keyframe' if verbose else f'{fa.ICON_FA_CAMERA}'
        space_required = imgui.calc_text_size(btn_text).x + 2 * imgui.get_style().frame_padding.x
        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        if verbose and space_required > imgui.get_content_region_avail().x:
            imgui.new_line()  # Move to next line if not enough space for button and tooltip
        if imgui.button(btn_text):
            # Request a clean render of the keyframe view to be saved
            camera_state = CameraState()
            screenshot_view: View = deepcopy(camera_state.current_view)  # type: ignore
            screenshot_view.c2w = keyframe.c2w
            screenshot_view.timestamp = keyframe.timestamp
            screenshot_view.camera = rescale_camera(
                keyframe.camera, *self._render_resolution,
                self._crop_to_resolution, self._maintain_pixel_aspect,
            )

            state.shared.screenshot(  # type: ignore
                screenshot_view,
                ViewerOutputState().selected_output,
                ColorMapState().dict,
            )
        tooltip("Take a preview screenshot from this keyframe's view")

        return delete_frame, clone_frame

    def _handle_export_file_dialog(self):
        """Handles the export file dialog state and exports keyframes to JSON file when ready."""
        if self._export_file_dialog is None:
            return

        # Check if dialog is ready
        if not self._export_file_dialog.ready(self._FILE_DIALOG_TIMEOUT):
            return

        # Get result
        result = self._export_file_dialog.result()
        self._export_file_dialog = None  # Clear the dialog

        if not result:
            # User canceled
            return

        filepath = Path(result).resolve()

        try:
            with open(filepath, 'w') as f:
                json.dump([kf.to_dict() for kf in self._keyframes], f, indent=2)
        except Exception as e:
            # Show error popup
            self._last_error_message = f'Failed to write file {filepath}: {e}'
            self._error_popup_open = True

    def _handle_import_file_dialog(self):
        """Handles the import file dialog state and imports keyframes from JSON file when ready."""
        if self._import_file_dialog is None:
            return

        # Check if dialog is ready
        if not self._import_file_dialog.ready(self._FILE_DIALOG_TIMEOUT):
            return

        # Get result
        results = self._import_file_dialog.result()
        self._import_file_dialog = None  # Clear the dialog

        if not results:
            # User canceled
            return

        filepath = Path(results[0]).resolve()  # result() returns a list

        try:
            with open(filepath, 'r') as f:
                parsed = json.load(f)

            if isinstance(parsed, list):
                self._add_keyframes_from_json(parsed)
                self._trajectory_modified = True
            else:
                self._last_error_message = 'Imported JSON is not a list of keyframes.'
                self._error_popup_open = True
        except Exception as e:
            self._last_error_message = f'Failed to import keyframes from {filepath}: {e}'
            self._error_popup_open = True

    def _add_keyframes_from_json(self, parsed: list[dict[str, Any]]):
        """Appends keyframes parsed from JSON, adding none if any keyframe is invalid."""
        self._keyframes.extend([Keyframe.from_dict(item) for item in parsed])
        if len(self._keyframes) > 0:
            self._keyframe_selected = self._keyframes[-1].identifier
