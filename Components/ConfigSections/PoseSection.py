"""Components/ConfigSections/PoseSection.py: Camera pose configuration section for the config window."""

from dataclasses import dataclass

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.Components.TensorInput import input_mat4
from ICGui.Controls import InputCallback
from ICGui.State.Volatile import GlobalState, CameraState, TimeState
from ICGui.util.Cameras import argmax_similarity
from ICGui.util.Enums import Action
from .Section import Section


@dataclass
class PoseSection(Section):
    name: str = f'{fa.ICON_FA_LOCATION_DOT} Pose'
    always_open: bool = False
    default_open: bool = False

    _show_advanced = False
    _matrices_as_text = False
    _last_pose_idx: int = -1

    def __post_init__(self):
        super().__post_init__()

        input_manager = GlobalState().input_manager
        input_manager.register_callback(
            'TOGGLE_GROUND_TRUTH',
            InputCallback(self.toggle_gt, continuous=False, interrupt_animation=True),
            'Toggle Ground Truth',
        )
        input_manager.register_callback(
            'JUMP_TO_CLOSEST_POSE',
            InputCallback(self.jump_to_closest, continuous=False, interrupt_animation=False),
            'Jump to closest dataset pose',
        )
        input_manager.register_callback(
            Action.NEXT_POSE,
            InputCallback(lambda: self.cycle_pose(1), continuous=False, interrupt_animation=False),
            'Next dataset pose',
        )
        input_manager.register_callback(
            Action.PREVIOUS_POSE,
            InputCallback(lambda: self.cycle_pose(-1), continuous=False, interrupt_animation=False),
            'Previous dataset pose',
        )
        input_manager.register_callback(
            Action.NEXT_DATASET_SPLIT,
            InputCallback(lambda: self.cycle_dataset_split(1), continuous=False, interrupt_animation=False),
            'Next dataset split',
        )
        input_manager.register_callback(
            Action.PREVIOUS_DATASET_SPLIT,
            InputCallback(lambda: self.cycle_dataset_split(-1), continuous=False, interrupt_animation=False),
            'Previous dataset split',
        )

    def _render(self):
        self._render_dataset_pose_selector()

        if imgui.button('Jump to closest dataset pose'):
            self.jump_to_closest()

        _, GlobalState().skip_animations = styled_toggle('Skip Animations', GlobalState().skip_animations)
        help_indicator('Skip camera animations when jumping to a new position.')

        changed, _ = styled_toggle('Show ground truth image', CameraState().render_gt)
        if changed:
            self.toggle_gt()
        help_indicator('Show the ground-truth image if available for the current pose, '
                       'scaled according to focal length / projection center.')

        imgui.begin_disabled(CameraState().render_gt)
        # TODO: timestamp should update according to the snapped-to GT view
        changed, snap_state = styled_toggle('Snap to ground truth', CameraState().render_gt or CameraState().snap_to_gt)
        if changed:
            CameraState().snap_to_gt = snap_state
        imgui.end_disabled()
        help_indicator('Lock the camera to the nearest ground truth pose. You can '
                       'still move the camera, but the view will only update, once '
                       'you\'re closer to another ground truth pose.')

        # Advanced settings (Direct Matrix Input)
        _, self._show_advanced = styled_toggle(
            'Show Advanced Options',
            self._show_advanced,
        )
        if self._show_advanced:
            self._render_advanced()

    def _render_dataset_pose_selector(self):
        global_state = GlobalState()
        input_manager = global_state.input_manager
        camera_state = CameraState()
        current_split = camera_state.dataset_split
        splits = global_state.shared.splits
        poses = camera_state.dataset_poses

        # Render Split Selector
        if imgui.begin_combo('Dataset Split', current_split):
            for split in splits:
                changed, val = imgui.selectable(split, split == current_split)
                if changed:
                    camera_state.dataset_split = split
                    pose = poses[split][0]
                    input_manager.control_scheme.ease_to(to_c2w=pose.c2w_numpy)
                    TimeState().time = pose.timestamp
                    self._last_pose_idx = 0
            imgui.end_combo()

        # Render Camera Pose Selector
        pose_idx = -1 if input_manager.control_scheme.has_moved else self._last_pose_idx
        if (imgui.begin_combo('##DatasetPoseSelector',
                              'Choose pose from dataset' if pose_idx == -1
                              else f'Pose {self._last_pose_idx}')):
            for i, cam in enumerate(poses[camera_state.dataset_split]):
                if imgui.selectable(f'Pose {i}', False)[0]:
                    input_manager.control_scheme.ease_to(to_c2w=cam.c2w_numpy)
                    TimeState().time = cam.timestamp
                    self._last_pose_idx = i
            imgui.end_combo()

    def _render_advanced(self):
        input_manager = GlobalState().input_manager

        _, self._matrices_as_text = styled_toggle('Show Matrices as Text', self._matrices_as_text)
        help_indicator('Show matrices as a multi-line text input, to support copy & paste. Changes are applied '
                       'on enter/focus loss of the text field. Valid formats include any set of 16 numbers, '
                       'separated by whitespace, commas or square brackets.')

        changed, c2w = input_mat4('Camera to World Transformation', input_manager.control_scheme.c2w,
                                  as_separate=not self._matrices_as_text)
        if changed:
            input_manager.control_scheme.c2w = c2w

        changed, w2c = input_mat4('World to Camera Transformation', input_manager.control_scheme.w2c,
                                  as_separate=not self._matrices_as_text)
        if changed:
            input_manager.control_scheme.w2c = w2c

    @staticmethod
    def toggle_gt():
        """Toggle the visibility of the ground truth."""
        gt_idx = CameraState().gt_idx
        CameraState().render_gt = not CameraState().render_gt

        # When toggling off, reset the current camera pose to the ground truth pose
        if gt_idx >= 0:
            pose = CameraState().dataset_poses[CameraState().dataset_split][gt_idx]
            GlobalState().input_manager.control_scheme.c2w = pose.c2w_numpy
            TimeState().time = pose.timestamp

    def jump_to_closest(self):
        """Jump to the closest pose in the dataset."""
        camera = GlobalState().input_manager.control_scheme
        camera_state = CameraState()
        cam_idx = argmax_similarity(camera.c2w, [
            pose.c2w_numpy for pose in camera_state.dataset_poses[camera_state.dataset_split]
        ])
        pose = camera_state.dataset_poses[camera_state.dataset_split][cam_idx]
        camera.ease_to(to_c2w=pose.c2w_numpy)
        TimeState().time = pose.timestamp
        self._last_pose_idx = cam_idx

    def cycle_pose(self, increment: int = 1):
        """Jump to the next pose in the dataset."""
        camera_state = CameraState()
        poses = camera_state.dataset_poses[camera_state.dataset_split]
        if not poses:
            return
        self._last_pose_idx = (self._last_pose_idx + increment) % len(poses)
        pose = poses[self._last_pose_idx]
        GlobalState().input_manager.control_scheme.ease_to(to_c2w=pose.c2w_numpy)
        TimeState().time = pose.timestamp

    def cycle_dataset_split(self, increment: int = 1):
        """Switch to the next dataset split."""
        camera_state = CameraState()
        global_state = GlobalState()

        splits = global_state.shared.splits
        if not splits:
            return
        current_idx = splits.index(camera_state.dataset_split)
        new_idx = (current_idx + increment) % len(splits)

        camera_state.dataset_split = splits[new_idx]
        self._last_pose_idx = 0
        if camera_state.dataset_poses[camera_state.dataset_split]:
            pose = camera_state.dataset_poses[camera_state.dataset_split][0]
            global_state.input_manager.control_scheme.ease_to(to_c2w=pose.c2w_numpy)
            TimeState().time = pose.timestamp
