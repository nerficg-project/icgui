"""Backend/SDL3Window.py: Backend proving windowing support ."""

import ctypes
from pathlib import Path
from typing import Any, Callable

import OpenGL.GL as gl
from imgui_bundle import imgui
from imgui_bundle.python_backends.sdl3_backend import SDL3Renderer
from sdl3 import *  # pylint: disable=redefined-builtin

from ICGui.util.Enums import CallbackType

import Framework
from Logging import Logger


class SDL3Window:
    WM_CLASS = 'NeRFICG'  # The window class name of the application (used e.g. for grouping related windows in the OS)
    CURSOR_SHAPE_MAP = {
        imgui.MouseCursor_.none: SDL_SYSTEM_CURSOR_CROSSHAIR,
        imgui.MouseCursor_.arrow: SDL_SYSTEM_CURSOR_DEFAULT,
        imgui.MouseCursor_.text_input: SDL_SYSTEM_CURSOR_TEXT,
        imgui.MouseCursor_.resize_all: SDL_SYSTEM_CURSOR_MOVE,
        imgui.MouseCursor_.resize_ns: SDL_SYSTEM_CURSOR_NS_RESIZE,
        imgui.MouseCursor_.resize_ew: SDL_SYSTEM_CURSOR_EW_RESIZE,
        imgui.MouseCursor_.resize_nesw: SDL_SYSTEM_CURSOR_NESW_RESIZE,
        imgui.MouseCursor_.resize_nwse: SDL_SYSTEM_CURSOR_NWSE_RESIZE,
        imgui.MouseCursor_.hand: SDL_SYSTEM_CURSOR_POINTER,
        imgui.MouseCursor_.wait: SDL_SYSTEM_CURSOR_WAIT,
        imgui.MouseCursor_.progress: SDL_SYSTEM_CURSOR_PROGRESS,
        imgui.MouseCursor_.not_allowed: SDL_SYSTEM_CURSOR_NOT_ALLOWED,
    }

    """Interface for interacting with the SDL3 ImGui Backend."""
    def __init__(self, window_name: str, initial_window_dimensions: tuple[int, int], *,
                 clear: bool = False, vsync: bool = True):
        """Creates an SDL3 window instance, which can be used as a context manager to render content to a window
        Args:
            window_name: Name of the window
            initial_window_dimensions: Initial window dimensions
            clear: Should the background be cleared each frame?
            vsync: Whether VSync should be requested. If VSync is not available, a warning is shown.
        """
        self.callbacks: dict[CallbackType, list[Callable]] = {}
        self.cursor: SDL_Cursor | None = None
        self.cursor_id: int = -2
        self.vsync: bool = vsync
        self.window: SDL_Window | None = None
        self.window_dimensions: tuple[ctypes.c_int, ctypes.c_int] = ctypes.c_int(initial_window_dimensions[0]), \
            ctypes.c_int(initial_window_dimensions[1])
        self._title: str = window_name
        self._clear: bool = clear
        self._gl_context: Any = None
        self._impl: SDL3Renderer | None = None

        self.mouse_captured = False
        self._mouse_delta: tuple[int, int] = (0, 0)

    @property
    def size(self) -> tuple[int, int]:
        """Returns the size of the window as (width, height)."""
        return self.window_dimensions[0].value, self.window_dimensions[1].value

    @property
    def is_fullscreen(self) -> bool:
        """Returns whether the window is currently in fullscreen mode."""
        return bool(SDL_GetWindowFlags(self.window) & SDL_WINDOW_FULLSCREEN)

    @property
    def mouse_delta(self) -> tuple[float, float]:
        """Returns the relative mouse movement since the last frame."""
        return self._mouse_delta

    def register_callback(self, callback_type: CallbackType, callback: Callable) -> None:
        """Registers a callback for the given callback type."""
        if callback_type not in self.callbacks:
            self.callbacks[callback_type] = []
        self.callbacks[callback_type].append(callback)

    def set_mouse_capture(self, capture: bool):
        """Captures the mouse cursor to the current window."""
        self.mouse_captured = capture
        self._mouse_delta = (0, 0)
        SDL_SetWindowRelativeMouseMode(self.window, capture)

    def prepare_frame(self) -> bool:
        """Called before each frame is rendered. If false is returned, the GUI should be closed."""
        running = True
        event = SDL_Event()

        while SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == SDL_EVENT_QUIT:
                running = False
                break
            elif event.type == SDL_EVENT_WINDOW_RESIZED:
                SDL_GetWindowSizeInPixels(self.window, ctypes.byref(self.window_dimensions[0]),
                                          ctypes.byref(self.window_dimensions[1]))
                for callback in self.callbacks.get(CallbackType.RESIZE, []):
                    if not callable(callback):
                        Logger.log_warning(f'Invalid callback {callback} for type {CallbackType.RESIZE}.')
                        continue
                    # Call the resize callback with the new window dimensions
                    callback(self.window_dimensions[0].value, self.window_dimensions[1].value)
            elif event.type == SDL_EVENT_MOUSE_MOTION and self.mouse_captured:
                self._mouse_delta = (
                    self._mouse_delta[0] + event.motion.xrel,
                    self._mouse_delta[1] + event.motion.yrel,
                )

            self._impl.process_event(event)
        self._impl.process_inputs()
        imgui.new_frame()

        # Ignore imgui mouse inputs when the mouse is captured
        if self.mouse_captured:
            imgui.get_io().config_flags |= imgui.ConfigFlags_.no_mouse
        else:
            imgui.get_io().config_flags &= ~imgui.ConfigFlags_.no_mouse

        if self._clear:
            gl.glClearColor(1.0, 1.0, 1.0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        return running

    def finalize_frame(self):
        """Finalize and draw the frame."""
        cursor_shape = imgui.get_mouse_cursor()
        if cursor_shape != self.cursor_id:
            self.cursor_id = cursor_shape
            self.cursor = SDL_CreateSystemCursor(self.CURSOR_SHAPE_MAP.get(self.cursor_id, SDL_SYSTEM_CURSOR_DEFAULT))
            SDL_SetCursor(self.cursor)

        imgui.render()
        self._impl.render(imgui.get_draw_data())

        SDL_GL_SwapWindow(self.window)

    def toggle_fullscreen(self):
        """Toggles the fullscreen mode of the window."""
        SDL_SetWindowFullscreen(self.window, not self.is_fullscreen)

    def resize_window(self, width: int, height: int):
        """Resizes the window to the given dimensions."""
        SDL_RestoreWindow(self.window)  # Ensure the window is not maximized before resizing
        SDL_SetWindowSize(self.window, width, height)

    def _setup_sdl(self):
        os.environ['SDL_VIDEO_X11_WMCLASS'] = self.WM_CLASS
        os.environ['SDL_VIDEO_WAYLAND_WMCLASS'] = self.WM_CLASS

        # Set up SDL3
        if not SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS):
            raise Framework.GUIError(f'Failed to initialize SDL3: {SDL_GetError().decode("utf-8")}')

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

        self.window = SDL_CreateWindow(
            self.title.encode('utf-8'),
            self.window_dimensions[0],
            self.window_dimensions[1],
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE,
        )
        if self.window is None:
            raise Framework.GUIError(f'Failed to create SDL3 window: {SDL_GetError().decode("utf-8")}')

        self._gl_context = SDL_GL_CreateContext(self.window)
        if self._gl_context is None:
            raise Framework.GUIError(f'Failed to create OpenGL context: {SDL_GetError().decode("utf-8")}')
        SDL_GL_MakeCurrent(self.window, self._gl_context)

    def _setup_imgui(self):
        imgui.create_context()
        self._impl = SDL3Renderer(self.window)

    def _enable_vsync(self):
        # Enable VSync if requested
        if self.vsync:
            if SDL_GL_SetSwapInterval(-1):
                Logger.log_info('GUI: Running with adaptive VSync enabled')
            elif SDL_GL_SetSwapInterval(1):
                Logger.log_info('GUI: Running with VSync enabled')
            else:
                Logger.log_warning(f'GUI: Failed to enable vsync: {SDL_GetError().decode("utf-8")}')
        else:
            if not SDL_GL_SetSwapInterval(0):
                Logger.log_warning(f'GUI: Failed to disable vsync: {SDL_GetError().decode("utf-8")}')

    def _set_application_icon(self):
        img_path = Path('resources/icg_logo.bmp')
        if not img_path.exists():
            Logger.log_warning(f'Application icon missing or inaccessible at: {img_path}')
            return

        application_icon = SDL_LoadBMP(str(img_path).encode('utf-8'))
        if application_icon is None:
            Logger.log_warning(f'Failed to load application icon: {SDL_GetError().decode("utf-8")}')
            return

        if not SDL_SetWindowIcon(self.window, application_icon):
            Logger.log_warning(Framework.GUIError(f'Failed to set application icon: {SDL_GetError().decode("utf-8")}'))
            return

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        SDL_SetWindowTitle(self.window, value.encode('utf-8'))

    def __enter__(self):
        """Initializes the rendering backend."""
        self._setup_sdl()
        self._enable_vsync()
        self._setup_imgui()
        self._set_application_icon()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Does any necessary cleanup after the GUI is closed."""
        self._impl.shutdown()
        SDL_GL_DestroyContext(self._gl_context)
        SDL_DestroyWindow(self.window)
        SDL_Quit()
