# -- coding: utf-8 --

"""Applications/ViewerGUI.py: Implements the main GUI, showing the model output with GUI elements overlayed."""

from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import imgui
from platformdirs import user_config_dir

from ICGui.Backends import BaseBackend, SDL2Backend
from ICGui.GuiComponents import BaseGuiComponent, \
    InfoComponent, \
    ViewerConfiguration, \
    CameraConfiguration, \
    RenderExtrasConfiguration, \
    TrainingDonePopup, \
    CloseWhileTrainingPopup
from ICGui.GuiControls import GuiCamera
from ICGui.GuiConfig import LaunchConfig, NeRFConfig
from ICGui.ModelRunners import ModelState
from ICGui.Viewers import BaseViewer, SDL2Viewer
from ICGui.Viewers.ScreenshotUtils import getScreenshotPath

from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Logging import Logger

try:
    # pylint: disable=import-outside-toplevel
    from matplotlib import font_manager
except ImportError:
    font_manager = None


class GuiKWArgs(TypedDict, total=False):
    """Keyword arguments accepted by the GUI"""
    width: int
    height: int
    resolution_factor: float


class ViewerGUI:  # pylint: disable = too-few-public-methods
    """Main GUI class for the viewing NerfICG models"""

    # Store CONFIG_PATH as class variable to prevent it from being garbage collected;
    #   imgui does not hold its own reference, resulting in undefined behavior otherwise
    CONFIG_PATH = str(Path(user_config_dir('NerfICG', 'TUBS-ICG',
                                           ensure_exists=True)) / 'imgui.ini').encode()

    def __init__(self, config: LaunchConfig, *, width: int = None, height: int = None, resolution_factor: float = None,
                 dataset_poses: dict[str, list[CameraProperties]] = None, dataset_camera: BaseCamera = None,
                 bbox_size: float, shared_state: ModelState = None, **_):
        self.config = config
        BaseGuiComponent.gui_config = config
        BaseGuiComponent.save_window_positions = config.save_window_positions

        self._dataset_poses = dataset_poses
        self._dataset_camera = dataset_camera
        self._dataset_bbox_size = bbox_size
        self._window_size: tuple[int, int] = (width or self.config.initial_resolution[0] or 1280,
                                              height or self.config.initial_resolution[1] or 720)
        self._resolution_factor: float = resolution_factor or self.config.resolution_factor or 1.0

        self._backend: BaseBackend = SDL2Backend('NerfICG', self._window_size,
                                                 vsync=self.config.vsync, resize_callback=self._updateSize)

        self._shared_state: ModelState = shared_state

        self._training = self._shared_state.is_training
        self._framerate_model: float = 0.0
        self._framerate_model_std: float = 0.0
        self._shared_state.window_size = self._window_size

        self._setupImGui()

    def _setupImGui(self):
        """Sets up the ImGui context and initializes the GUI components"""
        imgui.create_context()
        imgui.get_io().display_size = self._window_size
        imgui.get_io().ini_file_name = self.CONFIG_PATH
        self._setupFont()

        self._nerf_config = NeRFConfig(
            state=self._shared_state,
            window_size=self._window_size,
            dataset_poses=self._dataset_poses,
            dataset_camera=self._dataset_camera,
            getTrueResolution=self._backend.getTrueResolution,
            resizeWindowCallback=lambda w, h: self._backend.resizeWindow(w, h) or self._updateSize(w, h),
        )
        self._nerf_config.resolution_scaling = NeRFConfig.ResolutionScaleConfig(
            parent=self._nerf_config,
            factor=self._resolution_factor,
            max_adaptive_scale=self._resolution_factor
        )
        self._nerf_config.camera = NeRFConfig.CameraConfig(
            parent=self._nerf_config,
            current=deepcopy(self._dataset_camera),
            dataset_split=self._shared_state.gt_split,
            reset_to_dataset_nearfar=self.config.dataset_near_far,
        )
        self._nerf_config.screenshot = NeRFConfig.ScreenshotConfig()

        self._viewer: BaseViewer = SDL2Viewer(self._backend, self._nerf_config,
                                              self._resolution_factor)

        self._gui_camera = GuiCamera(self._nerf_config.camera.properties.w2c,
                                     self._backend,
                                     travel_speed=0.05 * self._dataset_bbox_size,
                                     get_key_mapping=self._backend.getKeyMapping,
                                     cuda=False)
        self._nerf_config.gui_camera = self._gui_camera

        self._info_component = InfoComponent(config=self._nerf_config)
        self._viewer_configuration = ViewerConfiguration(config=self._nerf_config)
        self._camera_configuration = CameraConfiguration(config=self._nerf_config)
        self._extras_configuration = RenderExtrasConfiguration(config=self._nerf_config)
        self._training_popup = TrainingDonePopup()
        self._close_popup = CloseWhileTrainingPopup()

    def _setupFont(self):
        """Sets up the font for the GUI"""
        if font_manager is not None:
            font_file = font_manager.findfont(font_manager.FontProperties(family=self.config.font_family,
                                                                          style='normal'),
                                              fontext='ttf', fallback_to_default=True)

            imgui.get_io().fonts.add_font_from_file_ttf(font_file, self.config.font_size)
            imgui.get_io().font_global_scale = 1
        else:
            Logger.logInfo('Font support not enabled, missing matplotlib dependency.')
            imgui.get_io().fonts.add_font_default()
            imgui.get_io().font_global_scale = 1.5

    def _updateModelParams(self, dt: float):
        """Updates the shared model parameters based on the current GUI state"""
        self._nerf_config.camera.update(self._gui_camera.c2w, dt)
        self._shared_state.camera = self._nerf_config.camera.render_cam
        self._shared_state.renderGt(self._nerf_config.camera.gt_idx, self._nerf_config.camera.dataset_split)

        # Render to full res texture if adaptive resolution scaling to avoid flicker
        resolution_factor = 1.0 if self._nerf_config.resolution_scaling.adaptive \
            else self._nerf_config.resolution_scaling.factor
        if resolution_factor != self._resolution_factor:
            self._resolution_factor = resolution_factor
            self._viewer.resize((self._window_size[0], self._window_size[1]), self._resolution_factor)

        # Switch from training to not training
        if not self._shared_state.is_training and self._training:
            self._training_popup.open()
            self._close_popup.close()
        self._training = self._shared_state.is_training

    def _updateNerfFrame(self):
        """Grabs a new frame from the render process if available and displays it"""
        frame = self._shared_state.frame

        if frame is None:
            return

        if frame['type'] == 'render':
            self._viewer.setMode('scaled')
            self._viewer.draw(frame, self._viewer_configuration.selected_output,
                              self._viewer_configuration.selected_colormap,
                              interpolate=self._nerf_config.resolution_scaling.adaptive)
        else:
            self._viewer.setMode('full')
            self._viewer.draw(frame, self._viewer_configuration.selected_output,
                              self._viewer_configuration.selected_colormap)

        self._viewer_configuration.updateOutputChoices(frame)
        self._framerate_model = frame['fps']
        self._framerate_model_std = frame['fps_std']
        self._nerf_config.resolution_scaling.update(frame['fps_last'])

    def _updateSize(self, width: int, height: int):
        """Resize callback used to update the width and height in the backend"""
        self._window_size = (width, height)
        self._viewer.resize((width, height))
        self._nerf_config.resize(self._window_size[0], self._window_size[1])

        self._shared_state.window_size = self._window_size

    def _render(self):
        """Renders the ImGui components"""
        if self._gui_camera.tile_windows:
            BaseGuiComponent.tileWindows()
        self._info_component.render(model_framerate=self._framerate_model, framerate_std=self._framerate_model_std,
                                    training_iter=self._shared_state.training_iteration if self._training else None)
        self._viewer_configuration.render()
        self._camera_configuration.render()
        self._extras_configuration.render(extras=self._viewer.extras)
        self._training_popup.render()

        imgui.render()
        imgui.end_frame()

    def _screenshot(self):
        """Takes a screenshot if the GUI camera is set to take one"""
        if not (self._gui_camera.screenshot or self._nerf_config.screenshot.should_take):
            return
        if self._nerf_config.screenshot.include_extras:
            self._viewer.saveScreenshot(getScreenshotPath('with-extras_'))
            return

        self._shared_state.screenshot(self._nerf_config.camera.screenshot_cam,
                                      self._viewer_configuration.selected_output,
                                      self._viewer_configuration.selected_colormap)

    def _updateRunnerConfig(self):
        """Updates the runner configuration based on the current GUI state"""
        if config := self._shared_state.config_options:
            self._viewer_configuration.updateConfigs(config)

    def run(self):
        """Runs the GUI loop as long as the backend reports that the window is open."""
        with self._backend as backend:
            self._viewer.initialize()
            running = True
            while running:
                dt = imgui.get_io().delta_time

                # Close the GUI if the window is closed and no training is occurring
                window_open = backend.beforeFrame()
                if not window_open and not self._training:
                    running = False

                # But if training is active, show a popup instead
                if not window_open and self._training:
                    self._close_popup.open()
                self._close_popup.render()
                if self._close_popup.response == self._close_popup.Response.CLOSE:
                    running = False
                if self._close_popup.response == self._close_popup.Response.TERMINATE:
                    self._shared_state.terminateTraining()
                    running = False

                self._updateRunnerConfig()
                self._gui_camera.handleInputs(dt)

                self._updateModelParams(dt)
                self._updateNerfFrame()

                # Update in-memory references to the current GUI state
                self._render()

                # Render the frame
                backend.clearFrame()
                self._viewer.render()
                self._viewer.renderExtras(self._extras_configuration.extras_enabled,
                                          self._nerf_config.camera.render_cam if not self.config.synchronize_extras
                                          else None)
                self._screenshot()
                backend.afterFrame()
