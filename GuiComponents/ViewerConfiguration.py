# -- coding: utf-8 --

"""GuiComponents/ViewerConfiguration.py: Allows configuring the Model Viewer from within the GUI."""

import inspect
from typing import Any

import imgui
import torch
from sdl2 import SDL_SCANCODE_F3, SDL_SCANCODE_V

from ICGui.GuiComponents.Base import GuiComponent
from ICGui.GuiConfig import NeRFConfig
from Visual.ColorMap import ColorMap


class ViewerConfiguration(GuiComponent):
    """Informational component for the GUI, showing FPS and hotkeys."""
    _HOTKEYS = [SDL_SCANCODE_F3, SDL_SCANCODE_V]
    _IGNORED_KEYS = ['total_samples', 'fps', 'fps_std', 'type']
    _DEPTH_COLOR_MAPS = ['Grayscale'] + [member[0]
                                         for member in inspect.getmembers(ColorMap, predicate=inspect.isfunction)
                                         if member[0].isupper()]

    def __init__(self, config: NeRFConfig):
        super().__init__('Viewer Config', ViewerConfiguration._HOTKEYS, config=config,
                         closable=True, default_open=False)
        self._config: NeRFConfig = config
        self._renderer_configs: list[tuple[str, type, Any]] = []
        self._output_choices: list[str] = ['rgb']
        self._selected_output: int = 0
        self._selected_colormap: int = 0
        self._show_renderer_config: bool = False

    def _renderResolutionInput(self):
        changed, value = imgui.input_int2('Resolution', *self._config.getTrueResolution(),
                                          flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        if changed:
            resolution = (
                max(1, value[0]),
                max(1, value[1])
            )
            self._config.resizeWindowCallback(*resolution)

        # Resolution scaling
        changed, resolution_factor = imgui.slider_float(
            'Resolution Factor', self._config.resolution_scaling.factor,
            self._config.resolution_scaling.min_scale, self._config.resolution_scaling.max_scale,
        )

        if changed:
            self._config.resolution_scaling.factor = max(self._config.resolution_scaling.min_scale,
                                                         min(self._config.resolution_scaling.max_scale,
                                                             resolution_factor))
            self._config.camera.recalculate()

        # Adaptive resolution scaling
        changed, self._config.resolution_scaling.adaptive = imgui.checkbox('Adaptive Resolution Scaling',
                                                                           self._config.resolution_scaling.adaptive)
        if imgui.is_item_hovered():
            imgui.set_tooltip('Automatically adjusts the resolution factor to maintain a target FPS.')

        # Recalculate camera if adaptive resolution scaling is toggled
        if changed:
            self._config.camera.recalculate()

        if self._config.resolution_scaling.adaptive:
            changed, target_fps = imgui.input_float('Target FPS', self._config.resolution_scaling.target_fps,
                                                    step=1.0, step_fast=5.0, format='%.0f',
                                                    flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if changed:
                self._config.resolution_scaling.target_fps = max(1.0, target_fps)

            _, self._config.resolution_scaling.max_adaptive_scale = imgui.slider_float(
                'Max Scale',
                self._config.resolution_scaling.max_adaptive_scale,
                self._config.resolution_scaling.min_scale, self._config.resolution_scaling.max_scale
            )

            _, dampening = imgui.slider_float(
                'Dampening',
                1.0 - self._config.resolution_scaling.dampening,
                0.0, 1.0
            )
            self._config.resolution_scaling.dampening = 1.0 - dampening

    def _renderScreenshotSettings(self):
        """Renders the screenshot resolution override options and screenshot button."""
        _, override_resolution = imgui.checkbox('Override Screenshot Resolution',
                                                self._config.screenshot.resolution_override is not None)
        if imgui.is_item_hovered():
            imgui.set_tooltip('! Incompatible with screenshots containing rendered extras !\n'
                              '! Large resolutions may result in the rendering process crashing !')

        if override_resolution:
            if self._config.screenshot.resolution_override is None:
                self._config.screenshot.resolution_override = self._config.getTrueResolution()

            _, new_values = imgui.input_int2('Screenshot Resolution Override',
                                             *self._config.screenshot.resolution_override)
            self._config.screenshot.resolution_override = (max(1, new_values[0]), max(1, new_values[1]))
        else:
            self._config.screenshot.resolution_override = None

        if imgui.button('Take Screenshot'):
            self._config.screenshot.take()

    def _renderOutputSelection(self):
        _, self._selected_output = imgui.combo('Model Output', self._selected_output, self._output_choices)
        if self._selected_output >= len(self._output_choices):
            self._selected_output = 0

        _, self._selected_colormap = imgui.combo('Color Map (for monochrome)', self._selected_colormap,
                                                 ViewerConfiguration._DEPTH_COLOR_MAPS)

    def _renderAdvancedRendererConfig(self):
        if len(self._renderer_configs) < 1:
            return

        if not self._show_renderer_config:
            if imgui.button('Show Advanced Renderer Configuration'):
                self._show_renderer_config = not self._show_renderer_config
            return

        imgui.text('Renderer Configuration')

        changes = {}
        for i, (key, value_type, value) in enumerate(self._renderer_configs):
            key_str = key.replace('Renderer.', '').replace('_', ' ').title()

            if value_type is bool:
                changed, value = imgui.checkbox(key_str, value)
            elif value_type is int:
                changed, value = imgui.input_int(key_str, value, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            elif value_type is float:
                changed, value = imgui.input_float(key_str, value, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            elif value_type is str:
                changed, value = imgui.input_text(key_str, value, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            else:
                raise ValueError(f'Unsupported renderer config type {value_type} for key {key}.')

            if changed:
                changes[key] = value
                self._renderer_configs[i] = (key, value_type, value)

        if changes:
            self._config.state.requestConfigChange(changes)

        if imgui.button('Hide Advanced Renderer Configuration'):
            self._show_renderer_config = not self._show_renderer_config

    # pylint: disable=arguments-differ
    def _render(self):
        """Renders the GUI component."""
        self._renderResolutionInput()
        self._renderScreenshotSettings()
        imgui.spacing()
        self._renderOutputSelection()
        imgui.separator()
        self._renderAdvancedRendererConfig()

    @staticmethod
    def isKeyValid(frame: dict[str, Any], key: str) -> bool:
        """Returns whether the given key is valid for the current frame."""
        if key in ViewerConfiguration._IGNORED_KEYS:
            return False
        if key not in frame.keys():
            return False
        if not isinstance(frame[key], torch.Tensor):
            return False
        if len(frame[key].shape) == 3 and frame[key].shape[2] > 4:
            return False
        if not 2 <= len(frame[key].shape) <= 3:
            return False

        return True

    def updateOutputChoices(self, frame: dict[str, Any]):
        """Updates the choices for the model output, e.g. ['rgb', 'depth']."""
        # Add new keys
        for key in frame.keys():
            if ViewerConfiguration.isKeyValid(frame, key) and key not in self._output_choices:
                self._output_choices.append(key)
            # Maintain current output choice while sorting
            current = self._output_choices[self._selected_output]
            self._output_choices.sort()
            self._selected_output = self._output_choices.index(current)

        # Remove keys that are no longer available or valid
        for idx, key in enumerate(self._output_choices):
            if not ViewerConfiguration.isKeyValid(frame, key):
                # Ensure the selected output is still valid
                if idx == self._selected_output:
                    self._selected_output = 0
                if idx < self._selected_output:
                    self._selected_output -= 1

                self._output_choices.remove(key)

    def updateConfigs(self, configs: list[tuple[str, Any]]):
        """Updates the runner configuration options."""
        for key, default_value in configs:
            match key.split('.')[0]:
                case 'Renderer':
                    self._renderer_configs.append((key, type(default_value), default_value))
                case _:
                    pass

    @property
    def selected_output(self) -> str:
        """Returns the currently selected model output."""
        return self._output_choices[self._selected_output]

    @property
    def selected_colormap(self) -> str:
        """Returns the currently selected depth color map."""
        return ViewerConfiguration._DEPTH_COLOR_MAPS[self._selected_colormap]
