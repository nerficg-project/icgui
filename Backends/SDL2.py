# -- coding: utf-8 --

"""Backends/SDL2.py: ImGui Backend Interface for the SDL2 backend."""

import ctypes
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import imgui
import OpenGL.GL as gl
from imgui.integrations.sdl2 import SDL2Renderer
from sdl2 import *  # pylint: disable=redefined-builtin

from ICGui.Backends.Base import BaseBackend
from ICGui.util.GuiAction import Action

import Framework
from Logging import Logger


class CustomBackend(BaseBackend):
    """Interface for interacting with the SDL2 ImGui Backend."""
    _KEYCODES: Mapping[str, int] = {
        'Shift': SDL_SCANCODE_LSHIFT,
    }

    _KEYMAP: Mapping[Action, tuple[int]] = {
        Action.FORWARD: (SDL_SCANCODE_W, SDL_SCANCODE_UP),
        Action.BACKWARD: (SDL_SCANCODE_S, SDL_SCANCODE_DOWN),
        Action.LEFT: (SDL_SCANCODE_A, SDL_SCANCODE_LEFT),
        Action.RIGHT: (SDL_SCANCODE_D, SDL_SCANCODE_RIGHT),
        Action.UP: (SDL_SCANCODE_R, SDL_SCANCODE_SPACE),
        Action.DOWN: (SDL_SCANCODE_LSHIFT, SDL_SCANCODE_LCTRL,
                      SDL_SCANCODE_RSHIFT, SDL_SCANCODE_RCTRL),
        Action.SPEED_UP: (SDL_SCANCODE_KP_PLUS, SDL_SCANCODE_EQUALS, SDL_SCANCODE_PAGEUP),
        Action.SPEED_DOWN: (SDL_SCANCODE_KP_MINUS, SDL_SCANCODE_MINUS, SDL_SCANCODE_PAGEDOWN),
        Action.SCREENSHOT: (SDL_SCANCODE_F2, SDL_SCANCODE_PRINTSCREEN),
        Action.FULLSCREEN: (SDL_SCANCODE_F11, ),
        Action.TILE_WINDOWS: (SDL_SCANCODE_F12, SDL_SCANCODE_HOME),
        Action.CYCLE_CAMERA_CONTROLS: (SDL_SCANCODE_LALT, SDL_SCANCODE_RALT, SDL_SCANCODE_END),
        Action.CANCEL: (SDL_SCANCODE_ESCAPE, ),
    }
    _keymap_warnings_emitted = set()

    def __init__(self, window_name: str, initial_window_dimensions: tuple[int, int], *,
                 vsync: bool = True, resize_callback: Callable[[int, int], None] = None):
        super().__init__(window_name, initial_window_dimensions, vsync=vsync, resize_callback=resize_callback)
        self.window: SDL_Window | None = None
        self._window_dimensions: tuple[ctypes.c_int, ctypes.c_int] = ctypes.c_int(initial_window_dimensions[0]), \
            ctypes.c_int(initial_window_dimensions[1])
        self.resolution_correction: tuple[int, int] = (0, 0)  # Workaround to match drawable area in resolution inputs
        self._resolution_correction_set = False
        self._context: Any = None
        self._impl: SDL2Renderer | None = None

        self._mouse_captured = False
        self._mouse_delta: tuple[int, int] = (0, 0)

    @property
    def window_size(self) -> tuple[int, int]:
        """Returns the size of the window as (width, height)."""
        return self._window_dimensions[0].value, self._window_dimensions[1].value

    def getTrueResolution(self) -> tuple[int, int]:
        """Returns the true resolution of the window as (width, height)."""
        return self._window_dimensions[0].value + self.resolution_correction[0], \
               self._window_dimensions[1].value + self.resolution_correction[1]

    def beforeFrame(self) -> bool:
        """Called before each frame is rendered. If false is returned, the GUI should be closed."""
        ret = True
        event = SDL_Event()
        while SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == SDL_QUIT:
                ret = False

            if event.type == SDL_WINDOWEVENT:
                if event.window.event == SDL_WINDOWEVENT_RESIZED:
                    if not self._resolution_correction_set:
                        # Calculate correction for resolution inputs if the first frame sent a resize event
                        width, height = self._window_dimensions[0].value, self._window_dimensions[1].value
                        SDL_GL_GetDrawableSize(self.window, ctypes.byref(self._window_dimensions[0]),
                                               ctypes.byref(self._window_dimensions[1]))
                        self.resolution_correction = (width - self._window_dimensions[0].value,
                                                      height - self._window_dimensions[1].value)

                    SDL_GL_GetDrawableSize(self.window, ctypes.byref(self._window_dimensions[0]),
                                           ctypes.byref(self._window_dimensions[1]))
                    if self.resize_callback is not None:
                        self.resize_callback(self._window_dimensions[0].value, self._window_dimensions[1].value)

            if event.type == SDL_MOUSEMOTION and self._mouse_captured:
                self._mouse_delta = (
                    self._mouse_delta[0] + event.motion.xrel,
                    self._mouse_delta[1] + event.motion.yrel,
                )

            self._impl.process_event(event)
        self._impl.process_inputs()

        self._resolution_correction_set = True  # Only the first frame is relevant for resolution correction
        return super().beforeFrame() and ret

    def clearFrame(self):
        """Called before finalizing the frame after the rendering loop."""
        gl.glClearColor(1.0, 1.0, 1.0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    def afterFrame(self):
        """Finalize and draw the frame."""
        super().afterFrame()
        self._impl.render(imgui.get_draw_data())
        SDL_GL_SwapWindow(self.window)

    def getKeyMapping(self, action: Action) -> tuple[int]:
        """Returns the key mapping used by PyImGui for the given action, as defined by SDL2."""
        if action not in self._KEYMAP:
            if action not in self._keymap_warnings_emitted:
                Logger.logWarning(f'No key mapping defined for action {action} by backend {__file__}.')
                self._keymap_warnings_emitted.add(action)
            return tuple()

        return self._KEYMAP[action]

    def getKeyCode(self, key: str) -> int | None:
        """Returns the key code for the given key, as defined by SDL2."""
        if key not in self._KEYCODES:
            Logger.logWarning(f'No key code defined for key {key} by backend {__file__}.')
            return None

        return self._KEYCODES[key]

    def captureMouse(self, capture: bool):
        """Captures the mouse cursor to the current window."""
        self._mouse_captured = capture
        self._mouse_delta = (0, 0)
        SDL_SetRelativeMouseMode(capture)

    def resizeWindow(self, width: int, height: int):
        """Resizes the window to the given dimensions."""
        SDL_SetWindowSize(self.window, width, height)
        self._window_dimensions = ctypes.c_int(width), ctypes.c_int(height)

    def toggleFullscreen(self):
        """Toggles the fullscreen mode of the window."""
        flags = SDL_GetWindowFlags(self.window)
        if flags & SDL_WINDOW_FULLSCREEN:
            SDL_SetWindowFullscreen(self.window, 0)
        else:
            SDL_SetWindowFullscreen(self.window, SDL_WINDOW_FULLSCREEN_DESKTOP)

    @property
    def mouse_delta(self) -> tuple[float, float]:
        """Returns the relative mouse movement since the last frame."""
        return self._mouse_delta

    def __enter__(self):
        """Initializes the rendering backend."""
        os.environ['SDL_VIDEO_X11_WMCLASS'] = 'NeRFICG'
        os.environ['SDL_VIDEO_WAYLAND_WMCLASS'] = 'NeRFICG'

        if SDL_Init(SDL_INIT_EVERYTHING) != 0:
            raise Framework.GUIError(f'SDL2: Failed to initialize SDL2: {SDL_GetError().decode("utf-8")}')

        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1)
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24)
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8)
        SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1)
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1)
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 8)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)

        SDL_SetHint(SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, b'1')
        SDL_SetHint(SDL_HINT_VIDEO_HIGHDPI_DISABLED, b'1')

        self.window = SDL_CreateWindow(
            self.window_name.encode('utf-8'),
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            self._window_dimensions[0].value,
            self._window_dimensions[1].value,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE,
        )

        if self.window is None:
            raise Framework.GUIError(f'SDL2: Failed to create window: {SDL_GetError().decode("utf-8")}')

        self._context = SDL_GL_CreateContext(self.window)
        if self._context is None:
            raise Framework.GUIError(f'SDL2: Failed to create OpenGL context: {SDL_GetError().decode("utf-8")}')

        SDL_GL_MakeCurrent(self.window, self._context)
        if self.vsync:
            if SDL_GL_SetSwapInterval(-1) == 0:
                Logger.logInfo('SDL2: Running with adaptive VSync enabled')
            elif SDL_GL_SetSwapInterval(1) == 0:
                Logger.logInfo('SDL2: Running with normal (non-adaptive) VSync enabled')
            else:
                Logger.logWarning(f'SDL2: Failed to enable vsync: {SDL_GetError().decode("utf-8")}')
        else:
            if SDL_GL_SetSwapInterval(0) != 0:
                Logger.logWarning(f'SDL2: Failed to disable vsync: {SDL_GetError().decode("utf-8")}')

        imgui.create_context()
        self._impl = SDL2Renderer(self.window)

        SDL_ClearError()
        try:
            if Path('resources/icg_logo.bmp').exists():
                logo = SDL_LoadBMP(b'resources/icg_logo.bmp')
                if SDL_GetError().decode('utf-8'):
                    raise Framework.GUIError(f'SDL2: Failed to load Application Icon: {SDL_GetError().decode("utf-8")}')

                SDL_SetWindowIcon(self.window, logo)
                if SDL_GetError().decode('utf-8'):
                    raise Framework.GUIError(f'SDL2: Failed to set Application Icon: {SDL_GetError().decode("utf-8")}')
        except Framework.GUIError as e:
            Logger.logWarning(f'Failed to load icon: {e}')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Does any necessary cleanup after the GUI is closed."""
        self._impl.shutdown()
        SDL_GL_DeleteContext(self._context)
        SDL_DestroyWindow(self.window)
        SDL_Quit()
