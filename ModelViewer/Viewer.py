"""ModelViewer/Viewer.py: Model Viewer that directly renders tensors from the GPU to the screen
without copying to and from the CPU, utilizing SDL, OpenGL, and CUDA."""

from math import ceil
from pathlib import Path
from typing import Mapping

import torch
import OpenGL.GL as gl
from cuda import cudart as cu
from OpenGL.GL.shaders import ShaderProgram
from sdl3 import *  # pylint: disable=redefined-builtin
from torchvision.utils import save_image

import Framework
from ICGui.Backend import CudaOpenGL, SDL3Window
from ICGui.State.Volatile import CameraState, GlobalState, ColorMapState
from .Overlays import BaseOverlay, UnitCube, CameraFrustums
from .Shaders import get_tensor_shader
from ICGui.util.ColorChannels import to_rgba

from Datasets.utils import View
from Logging import Logger


class ModelViewer:
    """Viewer that directly renders tensors from the GPU to the screen without copying them over to the GPU. Utilizes
    SDL3, OpenGL and CUDA."""
    def __init__(self, window: SDL3Window):
        """Initializes CUDA and registers the OpenGL texture. Requires SDL to be initialized beforehand."""
        self._texture_size: tuple[int, int] | None = None
        self._window: SDL3Window = window

        # Overlay settings
        self._gl_overlays: list[BaseOverlay] = []
        self._extras_view: View | None = None  # Stores the last rendered view for overlays rendering

        # Fields to be initialized during OpenGL initialization (see initializeOpenGL)
        self._model_shader_program: ShaderProgram | None = None
        self._model_vao_location: int | None = None
        self._model_tex_locations: dict[str, int] = {'model_output': 0, 'depth': 0, 'alpha': 0}
        self._texture_settings: dict[str, CudaOpenGL.TextureSettings] = {}

        # Fields to be initialized during CUDA initialization (see initializeCUDA)
        self._cuda_texture_locations: dict[str, cu.cudaGraphicsResource_t] = {}

    def initialize(self):
        """Initializes the viewer, called after the GUI is initialized. Initializing before will likely cause errors."""
        self._initialize_open_gl()
        self._initialize_cuda()

    def _initialize_open_gl(self):
        """Initializes OpenGL and the OpenGL texture used for rendering the frame."""
        if not SDL_Init(SDL_INIT_VIDEO):
            raise Framework.GUIError(f'SDL was not properly initialized before initializing the model viewer: '
                                     f'{SDL_GetError().decode("utf-8")}')

        self._texture_size = CameraState().texture_size
        self._model_shader_program = get_tensor_shader()
        self._model_vao_location = gl.glGenVertexArrays(1)

        for k in self._model_tex_locations.keys():
            self._texture_settings[k] = CudaOpenGL.TextureSettings(
                internal_format= gl.GL_RGBA32F if k == 'model_output' else gl.GL_R32F,
                pixel_format=gl.GL_RGBA if k == 'model_output' else gl.GL_RED,
                wrap_s=gl.GL_REPEAT, wrap_t=gl.GL_REPEAT,
            )
            self._model_tex_locations[k] = CudaOpenGL.generate_texture(self._texture_size, self._texture_settings[k])

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Register extra items to render
        self._gl_overlays.append(UnitCube())
        self._gl_overlays.append(CameraFrustums(initial_scale=0.15 * CameraState().bbox_size))

    def _initialize_cuda(self):
        """Initializes CUDA and registers the OpenGL texture as a CUDA image."""
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise Framework.GUIError(f'No CUDA-capable devices found: {err}')

        for k in self._model_tex_locations.keys():
            self._cuda_texture_locations[k] = CudaOpenGL.register_cuda_texture(self._model_tex_locations[k])

    @staticmethod
    def resize(window_size: tuple[int, int]):
        """Resizes the viewer to the given size."""
        gl.glViewport(0, 0, *window_size)

    def _resize_textures(self):
        """Resizes the internal texture to the current window size."""
        self._texture_size = CameraState().texture_size
        for k in self._model_tex_locations.keys():
            self._cuda_texture_locations[k] = CudaOpenGL.resize_texture(
                self._model_tex_locations[k],
                self._texture_size,
                self._texture_settings[k],
                self._cuda_texture_locations.get(k, None),
            )

    def set_output_texture_filtering(self, minify: int = None, magnify: int = None):
        """Sets the texture filtering mode for minification and magnification."""
        CudaOpenGL.set_texture_filtering(self._model_tex_locations['model_output'], minify, magnify)

    def render(self):
        """Renders the current stored frame to the screen."""
        gl.glUseProgram(self._model_shader_program)
        try:
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._model_tex_locations['model_output'])
            gl.glBindVertexArray(self._model_vao_location)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)

    def render_extras(self, extras_enabled: dict[str, bool], view: View | None = None):
        """Renders extras, such as the wireframe unit cube, to the screen."""
        if self._extras_view is None and view is None:
            return  # No camera available to render extras

        extra_params = {
            'modelTextureLocations': self._model_tex_locations,
            'viewportSize': GlobalState().window_size,
        }
        for extra in self._gl_overlays:
            if extras_enabled.get(extra.name, False):
                extra.render(self._extras_view if view is None else view, extra_params)

    def draw_to_texture(self, img: torch.Tensor, texture: str = 'model_output', channels: int = 4,
                        interpolate: bool = False):
        """Draws the given RGBA tensor to the screen. The tensor must be on the GPU. If interpolate = True, an
        incorrectly-sized tensor is bilinearly interpolated, otherwise it is cropped/padded."""
        if not isinstance(img, torch.Tensor):
            return
        if len(img.shape) != 3 or img.shape[2] != channels:
            Logger.log_warning(f'Model viewer received invalid tensor of shape {img.shape}, skipping frame')
            return

        target_width, target_height = CameraState().texture_size

        if img.shape != (target_height, target_width, channels):
            if interpolate:
                # Interpolate the tensor size
                img = torch.nn.functional.interpolate(
                    img.permute(2, 0, 1)[None], (target_height, target_width), mode='bilinear', align_corners=False
                )[0].permute(1, 2, 0)

            else:
                # If the tensor is too big, crop it to the target size
                img = img[:target_height, :target_width, :]

                # Pad the tensor with zeros if it is smaller than the window size
                # This can occur e.g. in stereo cameras (1 pixel difference) or when resizing the window
                if img.shape[0] < target_height or img.shape[1] < target_width:
                    pad_x = target_width - img.shape[1]
                    pad_y = target_height - img.shape[0]
                    img = torch.nn.functional.pad(
                        img.permute(2, 0, 1), (pad_x // 2, ceil(pad_x / 2), pad_y // 2, ceil(pad_y / 2)), mode='replicate'
                    ).permute(1, 2, 0)

        CudaOpenGL.draw_tensor_to_texture(img, self._cuda_texture_locations[texture], channels=channels)

    def draw(self, imgs: Mapping[str, torch.Tensor], color_mode: str, interpolate=False):
        """Draws the given tensor to the window, automatically determining the correct method to call based on the
        color mode (e.g. 'rgb' or 'depth'). If interpolate = True, an incorrectly-sized tensor is bilinearly
        interpolated, otherwise it is cropped/padded."""
        if self._texture_size != CameraState().texture_size:
            self._resize_textures()

        rgba = to_rgba(imgs, color_mode, ColorMapState().dict)
        self.draw_to_texture(rgba, 'model_output', interpolate=interpolate)

        # Also store depth, alpha and camera for rendering extras
        if 'depth' in imgs.keys():
            self.draw_to_texture(imgs['depth'], 'depth', 1, interpolate=interpolate)
        if 'alpha' in imgs.keys():
            self.draw_to_texture(imgs['alpha'], 'alpha', 1, interpolate=interpolate)
        if 'view' in imgs.keys():
            self._extras_view = imgs['view']

    def save_screenshot(self, path: Path):
        """Saves the current frame including extras to the user's screenshot directory, taken from the current frame
        buffer."""
        buffer = gl.glReadPixels(0, 0, *self._texture_size, gl.GL_RGB, gl.GL_FLOAT)
        tensor = torch.frombuffer(buffer, dtype=torch.float32)
        tensor = tensor.reshape(self._texture_size[1], self._texture_size[0], 3).permute(2, 0, 1).flip(1)

        save_image(tensor, path)
        Logger.log_info(f'Saved screenshot with extras to {path}')

    @property
    def overlays(self) -> list[BaseOverlay]:
        """Returns a list of extras that can be rendered."""
        return self._gl_overlays
