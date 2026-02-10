"""Backend/CudaOpenGL.py: Interop between CUDA and OpenGL to facilitate rendering of tensors directly
from the GPU to the screen."""

from dataclasses import dataclass

import torch
import OpenGL.GL as gl
from cuda import cudart as cu

import Framework


@dataclass
class TextureSettings:
    """Settings for the OpenGL texture used for rendering."""
    internal_format: int = gl.GL_RGBA32F
    pixel_format: int = gl.GL_RGBA
    dtype: int = gl.GL_FLOAT
    min_filter: int = gl.GL_LINEAR
    mag_filter: int = gl.GL_LINEAR
    wrap_s: int = gl.GL_CLAMP_TO_EDGE
    wrap_t: int = gl.GL_CLAMP_TO_EDGE


def generate_texture(size: tuple[int, int], settings: TextureSettings = None) -> int:
    if settings is None:
        settings = TextureSettings()
    try:
        texture_location = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_location)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, settings.wrap_s)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, settings.wrap_t)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, settings.min_filter)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, settings.mag_filter)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            settings.internal_format,
            size[0],
            size[1],
            0,
            settings.pixel_format,
            settings.dtype,
            None,
        )
    finally:
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_location


def register_cuda_texture(texture_location: int) -> cu.cudaGraphicsResource_t:
    err, cuda_texture_location = cu.cudaGraphicsGLRegisterImage(
        texture_location,
        gl.GL_TEXTURE_2D,
        cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
    )
    if err != cu.cudaError_t.cudaSuccess:
        raise Framework.GUIError(f'Could not register OpenGL texture with CUDA: {err}')
    return cuda_texture_location


def unregister_cuda_texture(cuda_texture_location: cu.cudaGraphicsResource_t) -> None:
    (err,) = cu.cudaGraphicsUnregisterResource(cuda_texture_location)
    if err != cu.cudaError_t.cudaSuccess:
        raise Framework.GUIError(f'Could not unregister OpenGL texture from CUDA: {err}')


def resize_texture(texture_location: int, size: tuple[int, int], settings: TextureSettings,
                   cuda_texture_location: cu.cudaGraphicsResource_t = None) -> None | cu.cudaGraphicsResource_t:
    if cuda_texture_location is not None:
        unregister_cuda_texture(cuda_texture_location)

    try:
        # Resize the texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_location)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            settings.internal_format,
            size[0],
            size[1],
            0,
            settings.pixel_format,
            settings.dtype,
            None,
        )
    finally:
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    if cuda_texture_location is not None:
        return register_cuda_texture(texture_location)
    return None


def draw_tensor_to_texture(img: torch.Tensor, texture_location: cu.cudaGraphicsResource_t, *, channels: int = 4):
    """Draws the given tensor to the screen. The tensor must be on the GPU, and is assumed
    to be in HWC format, with the correct width/height for the texture."""
    img = img.contiguous()

    (err,) = cu.cudaGraphicsMapResources(1, texture_location, cu.cudaStreamLegacy)
    if err != cu.cudaError_t.cudaSuccess:
        raise Framework.GUIError(f'CUDAError: Unable to map graphics resource: {err}')
    err, array_ptr = cu.cudaGraphicsSubResourceGetMappedArray(texture_location, 0, 0)
    if err != cu.cudaError_t.cudaSuccess:
        raise Framework.GUIError(f'CUDAError: Unable to get mapped array: {err}')
    (err,) = cu.cudaMemcpy2DToArrayAsync(
        array_ptr,
        0,
        0,
        img.data_ptr(),
        channels * 4 * img.shape[1],
        channels * 4 * img.shape[1],
        img.shape[0],
        cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        cu.cudaStreamLegacy,
    )
    if err != cu.cudaError_t.cudaSuccess:
        raise Framework.GUIError(f'CUDAError: Unable to copy from tensor to texture: {err}')

    (err,) = cu.cudaGraphicsUnmapResources(1, texture_location, cu.cudaStreamLegacy)
    if err != cu.cudaError_t.cudaSuccess:
        raise Framework.GUIError(f'CUDAError: Unable to unmap graphics resource: {err}')


def set_texture_filtering(texture_location: int, minify: int = None, magnify: int = None):
    """Sets the texture filtering mode for minification and magnification."""
    try:
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_location)
        if minify is not None:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, minify)
        if magnify is not None:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, magnify)
    finally:
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
