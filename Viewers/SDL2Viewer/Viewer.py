# -- coding: utf-8 --

"""SDL2Viewer/Viewer.py: Implementation of the Base Viewer that directly renders tensors from the GPU to the screen
without copying to and from the CPU, utilizing SDL2, OpenGL, and CUDA."""
from math import ceil, floor
from pathlib import Path
from typing import Literal, Mapping

import torch
import OpenGL.GL as gl
from cuda import cudart as cu
from OpenGL.GL.shaders import ShaderProgram
from sdl2 import *  # pylint: disable=redefined-builtin
from torchvision.utils import save_image

from ICGui.Viewers.SDL2Viewer.Extras import BaseGLExtra, UnitCube, CameraFrustrums
from ICGui.Viewers.SDL2Viewer.Shaders import getTensorShader
from ICGui.Viewers.Base import BaseViewer, BaseExtra
from ICGui.Viewers.utils import ViewerError, toRgba
from ICGui.Backends import BaseBackend

from Cameras.Base import BaseCamera
from Logging import Logger


class CustomViewer(BaseViewer):
    """Viewer that directly renders tensors from the GPU to the screen without copying them over to the GPU. Utilizes
    SDL2, OpenGL and CUDA."""
    def __init__(self, backend: BaseBackend, config: 'NeRFConfig', resolution_factor: float = 1.0):
        """Initializes CUDA and registers the OpenGL texture. Requires SDL to be initialized beforehand."""
        super().__init__(backend, config, resolution_factor)
        self._mode = 'scaled'
        self._dimensions: tuple[int, int] = backend.window_size
        self._texture_size_invalid: bool = False

        # Fields to be initialized during CUDA initialization
        self._cuda_texture_locations: dict[str, cu.Any | None] = {'model_output': None, 'depth': None, 'alpha': None}

        # Fields to be initialized during OpenGL initialization
        self._model_shader_program: ShaderProgram | None = None
        self._model_vao_location: int | None = None
        self._model_tex_locations: dict[str, int] = {'model_output': 0, 'depth': 0, 'alpha': 0}
        self._gl_extras: list[BaseGLExtra] = []
        self._extras_camera: BaseCamera | None = None

    def initialize(self):
        """Initializes the viewer, called after the GUI is initialized. Initializing before will likely cause errors."""
        self._initializeOpenGL()
        self._initializeCUDA()

    def _initializeOpenGL(self):
        """Initializes OpenGL and the OpenGL texture used for rendering the frame."""
        if SDL_Init(SDL_INIT_VIDEO) != 0:
            raise ViewerError(f'SDL2-Viewer: SDL was not properly initialized: '
                              f'{SDL_GetError().decode("utf-8")}')

        self._model_shader_program = getTensorShader()
        self._model_vao_location = gl.glGenVertexArrays(1)

        for k in self._model_tex_locations:
            self._model_tex_locations[k] = gl.glGenTextures(1)

            gl.glBindTexture(gl.GL_TEXTURE_2D, self._model_tex_locations[k])
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

            internal_format = gl.GL_RGBA32F if k == 'model_output' else gl.GL_R32F
            pixel_format = gl.GL_RGBA if k == 'model_output' else gl.GL_RED
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                internal_format,
                int(self._dimensions[0] * self._resolution_factor),
                int(self._dimensions[1] * self._resolution_factor),
                0,
                pixel_format,
                gl.GL_FLOAT,
                None,
            )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Register extra items to render
        self._gl_extras.append(UnitCube())
        self._gl_extras.append(
            CameraFrustrums(initial_scale=0.15 * (self.config.dataset_camera.far_plane
                                                  - self.config.dataset_camera.near_plane)))

    def _initializeCUDA(self):
        """Initializes CUDA and registers the OpenGL texture as a CUDA image."""
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise ViewerError(f'SDL2-Viewer: No CUDA-capable devices found: {err}')

        for k in self._cuda_texture_locations:  # pylint: disable=consider-using-dict-items
            err, self._cuda_texture_locations[k] = cu.cudaGraphicsGLRegisterImage(
                self._model_tex_locations[k],
                gl.GL_TEXTURE_2D,
                cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
            )
            if err != cu.cudaError_t.cudaSuccess:
                raise ViewerError(f'SDL2-Viewer: Could not register OpenGL texture with CUDA: {err}')

    def resize(self, window_size: tuple[int, int], resolution_factor: float | None = None):
        """Resizes the viewer to the given size and marks the internal texture to be updated on the next draw call.
        Resolution factor None (default) means no change to the resolution factor"""
        super().resize(window_size, resolution_factor)
        self._dimensions = window_size
        self._texture_size_invalid = True

        gl.glViewport(0, 0, self._dimensions[0], self._dimensions[1])

    def _resizeTextures(self):
        """Resizes the internal texture to the current window size."""
        for k in self._model_tex_locations:  # pylint: disable=consider-using-dict-items
            # Unregister the old texture from CUDA
            cu.cudaGraphicsUnregisterResource(self._cuda_texture_locations[k])

            # Resize the texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._model_tex_locations[k])

            internal_format = gl.GL_RGBA32F if k == 'model_output' else gl.GL_R32F
            pixel_format = gl.GL_RGBA if k == 'model_output' else gl.GL_RED
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                internal_format,
                self._dimensions[0] if self._mode == 'full' else int(self._dimensions[0] * self._resolution_factor),
                self._dimensions[1] if self._mode == 'full' else int(self._dimensions[1] * self._resolution_factor),
                0,
                pixel_format,
                gl.GL_FLOAT,
                None,
            )

            # Register the new texture with CUDA
            err, self._cuda_texture_locations[k] = cu.cudaGraphicsGLRegisterImage(
                self._model_tex_locations[k],
                gl.GL_TEXTURE_2D,
                cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
            )
            if err != cu.cudaError_t.cudaSuccess:
                raise ViewerError(f'SDL2-Viewer: Could not register OpenGL texture with CUDA: {err}')

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

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

    def renderExtras(self, extras_enabled: dict[str, bool], camera_override: BaseCamera | None = None):
        """Renders extras, such as the wireframe unit cube, to the screen."""
        if self._extras_camera is None and camera_override is None:
            return

        extra_params = {
            'nerfconfig': self.config,
            'modelTextureLocations': self._model_tex_locations,
            'viewportSize': self._dimensions,
        }
        for extra in self._gl_extras:
            if extras_enabled.get(extra.name, False):
                extra.render(self._extras_camera if camera_override is None else camera_override,
                             extra_params)

    def drawToTexture(self, img: torch.Tensor, texture: str = 'model_output', channels: int = 4,
                      interpolate: bool = False):
        """Draws the given RGBA tensor to the screen. The tensor must be on the GPU. If interpolate = True, an
        incorrectly-sized tensor is bilinearly interpolated, otherwise it is cropped/padded."""
        if not isinstance(img, torch.Tensor):
            return
        if len(img.shape) != 3 or img.shape[2] != channels:
            Logger.logWarning(f'SDL2-Viewer: Received invalid tensor of shape {img.shape}, skipping frame')
            return

        target_width = self._dimensions[0] if self._mode == 'full' \
            else int(self._dimensions[0] * self._resolution_factor)
        target_height = self._dimensions[1] if self._mode == 'full' \
            else int(self._dimensions[1] * self._resolution_factor)

        if img.shape != (target_height, target_width, 4):
            if interpolate:
                # Interpolate the tensor size
                img = img.permute(2, 0, 1).flip(2)
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), (target_height, target_width), mode='bilinear', align_corners=False
                ).squeeze(0).permute(1, 2, 0).flip(1).contiguous(memory_format=torch.contiguous_format)

            else:
                # If the tensor is too big, cut it down
                img = img[:target_height, :target_width, :]

                # Pad the tensor with zeros if it is smaller than the window size
                # This can occur in stereo cameras (1 pixel difference) or when resizing the window
                if img.shape[0] < target_height or img.shape[1] < target_width:
                    pad_x = target_width - img.shape[1]
                    pad_y = target_height - img.shape[0]
                    img = img.permute((2, 0, 1))
                    img = torch.nn.functional.pad(img, (floor(pad_x / 2), ceil(pad_x / 2),
                                                        floor(pad_y / 2), ceil(pad_y / 2),), mode='replicate')
                    img = img.permute((1, 2, 0))
                img = img.contiguous(memory_format=torch.contiguous_format)

        (err,) = cu.cudaGraphicsMapResources(1, self._cuda_texture_locations[texture], cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise ViewerError(f'SDL2-Viewer: Unable to map graphics resource: {err}')
        err, array_ptr = cu.cudaGraphicsSubResourceGetMappedArray(self._cuda_texture_locations[texture], 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise ViewerError(f'SDL2-Viewer: Unable to get mapped array: {err}')
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array_ptr,
            0,
            0,
            img.data_ptr(),
            channels * 4 * target_width,
            channels * 4 * target_width,
            target_height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise ViewerError(f'SDL2-Viewer: Unable to copy from tensor to texture: {err}')

        (err,) = cu.cudaGraphicsUnmapResources(1, self._cuda_texture_locations[texture], cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise ViewerError(f'SDL2-Viewer: Unable to unmap graphics resource: {err}')

    def draw(self, imgs: Mapping[str, torch.Tensor], color_mode: str, color_map: str = 'Grayscale',
             interpolate=False):
        """Draws the given tensor to the window, automatically determining the correct method to call based on the
        color mode (e.g. 'rgb' or 'depth'). If interpolate = True, an incorrectly-sized tensor is bilinearly
        interpolated, otherwise it is cropped/padded."""
        if self._texture_size_invalid:
            self._resizeTextures()
            self._texture_size_invalid = False

        rgba = toRgba(imgs, color_mode, color_map)
        self.drawToTexture(rgba, 'model_output', interpolate=interpolate)

        if 'depth' in imgs.keys():
            self.drawToTexture(imgs['depth'], 'depth', 1, interpolate=interpolate)
        if 'alpha' in imgs.keys():
            self.drawToTexture(imgs['alpha'], 'alpha', 1, interpolate=interpolate)
        if 'camera' in imgs.keys():
            self._extras_camera = imgs['camera']

    def saveScreenshot(self, path: Path):
        """Saves the current frame including extras to the user's screenshot directory, taken from the current frame
        buffer."""
        buffer = gl.glReadPixels(0, 0, self._dimensions[0], self._dimensions[1], gl.GL_RGB, gl.GL_FLOAT)
        tensor = torch.frombuffer(buffer, dtype=torch.float32)
        tensor = torch.reshape(tensor, (self._dimensions[1], self._dimensions[0], 3)).permute((2, 0, 1)).flip(1)

        save_image(tensor, path)
        Logger.logInfo(f'SDL2-Viewer: Saved screenshot with extras to {path}')

    def setMode(self, mode: Literal['full', 'scaled']):
        """Sets the mode to draw textures at full resolution or upscale them."""
        if self._mode != mode:
            self._texture_size_invalid = True
        self._mode = mode

    @property
    def extras(self) -> list[BaseExtra]:
        """Returns a list of extras that can be rendered."""
        return self._gl_extras
