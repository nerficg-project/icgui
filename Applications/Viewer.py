"""Applications/Viewer.py: Implements the main GUI, showing the model output with
GUI elements to control settings overlayed."""

from copy import deepcopy
from imgui_bundle import imgui, icons_fontawesome_6 as fa
from torch import multiprocessing

from ICGui.Backend import SDL3Window, FontManager
from ICGui.Components.Windows import WindowManager, ConfigWindow, ModalWindow
from ICGui.Controls import InputHandler
from ICGui.ModelViewer import ModelViewer
from ICGui.util.Screenshots import get_screenshot_path
from ICGui.State import LaunchConfig, SharedState, Directories
from ICGui.State.Volatile import GlobalState, CameraState, ResolutionScaleState, TimeState, ScreenshotState, \
    ViewerOutputState, ColorMapState, OverlayState
from ICGui.util.Enums import CallbackType, Action

from Datasets.utils import View

try:
    # pylint: disable=import-outside-toplevel
    from matplotlib import font_manager
except ImportError:
    font_manager = None


class Viewer:  # pylint: disable = too-few-public-methods
    """Main GUI class for viewing NerfICG models"""
    CLOSE_MODAL_TITLE = f'{fa.ICON_FA_DOOR_OPEN} Close Viewer?'
    TRAINING_FINISHED_TITLE = f'{fa.ICON_FA_FLAG_CHECKERED} Training Finished'

    def __init__(self, config: LaunchConfig, *, width: int = None, height: int = None, resolution_factor: float = None,
                 dataset_poses: dict[str, list[View]] = None, default_view: View = None, bbox_size: float,
                 shared_state: SharedState = None, **_):
        self.launch_config = config
        self._crash_modal_title = f'{fa.ICON_FA_HEART_CRACK} Crash Detected'
        self._shown_crash_message = False

        if width is not None and height is not None:
            self.launch_config.initial_resolution = [width, height]
        elif width is not None or height is not None:
            raise ValueError('Both width and height must be provided if one is specified.')

        if resolution_factor is not None:
            self.launch_config.resolution_factor = resolution_factor

        # Store shared state and initialize global state singletons
        self._shared_state: SharedState = shared_state
        GlobalState(
            shared=self._shared_state,
            launch_config=self.launch_config,
        )
        CameraState(
            dataset_view=deepcopy(default_view),
            dataset_camera=deepcopy(default_view.camera),
            dataset_poses=dataset_poses,
            bbox_size=bbox_size
        )

        # Store previous training state to detect transition from training to inference
        self._was_training = self._shared_state.is_training

        # Set up window
        self._backend_window: SDL3Window = SDL3Window(
            window_name='NerfICG',
            initial_window_dimensions=GlobalState().window_size,
            vsync=self.launch_config.vsync
        )
        self._backend_window.register_callback(CallbackType.RESIZE, self.resize)
        GlobalState().backend_window = self._backend_window

        # Set up viewer and input handler components
        # TODO: Allow the ModelViewer to be replaced with a ViewerWindow
        self._viewer: ModelViewer = ModelViewer(self._backend_window)
        self._input_handler = InputHandler(self._backend_window, CameraState().current_view._c2w,  # FIXME: Use public method
                                           camera_travel_speed=0.05 * CameraState().bbox_size)
        GlobalState().viewer = self._viewer
        GlobalState().input_manager = self._input_handler
        OverlayState().available_overlays = self._viewer.overlays

        # Set up GUI components
        self._setup_imgui()

    def _setup_imgui(self):
        """Sets up the ImGui context and initializes the GUI components"""
        imgui.create_context()
        imgui.get_io().display_size = GlobalState().window_size  # type: ignore
        imgui.get_io().set_ini_filename(str(Directories.USER_CONFIG_DIR / 'imgui.ini'))
        self.fonts = FontManager.set_font(self.launch_config.font.name, self.launch_config.font.size)

        self.window_manager: WindowManager = WindowManager()
        self.window_manager.add_window(ConfigWindow)

    def _send_frame_request(self, dt: float):
        """Sends the current GUI state to the renderer process to request a new frame"""
        CameraState().update(self._input_handler.control_scheme.c2w)
        TimeState().update(dt)
        self._shared_state.view = CameraState().current_view
        self._shared_state.render_gt(CameraState().gt_idx, CameraState().dataset_split)

    def _receive_frame(self):
        """Grabs a new frame from the render process if available and displays it"""
        frame = self._shared_state.frame

        if frame is None:
            return

        ViewerOutputState().update_output_choices(frame)
        if ViewerOutputState().selected_output in frame:
            ColorMapState().update(frame[ViewerOutputState().selected_output])

            if frame['type'] == 'render':
                self._viewer.draw(frame, ViewerOutputState().selected_output,
                                  interpolate=ResolutionScaleState().adaptive)
            else:
                # Don't use resolution scaling for GT frames
                self._viewer.draw(frame, ViewerOutputState().selected_output)

        GlobalState().model_framerate = frame['fps']
        GlobalState().model_framerate_std = frame['fps_std']
        GlobalState().model_frametime = frame['frametime_mean']
        GlobalState().model_frametime_std = frame['frametime_std']
        ResolutionScaleState().adapt(frame['fps_last'])

    def _render(self):
        """Renders the ImGui components"""
        self.window_manager.render()

        imgui.render()
        imgui.end_frame()

    def _screenshot(self):
        """Takes a screenshot if requested by the user"""
        # Check if the hotkey is pressed, or the button in the GUI has been activated
        screenshot_requested = (self._input_handler.is_pressed(Action.SCREENSHOT)
                                or ScreenshotState().should_take)
        if not screenshot_requested:
            return

        if ScreenshotState().include_overlays:
            # Save the current frame with all rendered extras from the viewer
            self._viewer.save_screenshot(get_screenshot_path('overlays'))
        else:
            # Request a clean render of the current frame to be saved
            self._shared_state.screenshot(
                CameraState().screenshot_view,
                ViewerOutputState().selected_output,
                ColorMapState().dict,
            )

    def _show_training_done_modal(self) -> bool:
        """If training has finished, show a modal window to inform the user."""
        retval = True

        # Detect switch from training to not training
        if not self._shared_state.is_training and self._was_training:
            self._was_training = self._shared_state.is_training

            # If training has finished, the close modal is no longer valid
            self.window_manager.close_modal_window(self.CLOSE_MODAL_TITLE)

            # Instead, open up a modal window to inform the user that training has finished
            training_done_modal = ModalWindow(
                title=self.TRAINING_FINISHED_TITLE,
                message='Training has successfully finished. Close the GUI or keep it open?',
                options=[
                    'Close',
                    'Keep Open',
                ],
                option_colors=[
                    'ERROR',
                    'DEFAULT',
                ],
                close_action=1,
            )
            self.window_manager.add_modal_window(training_done_modal)

        # If the close modal is open, handle its response
        if (close_response := self.window_manager.get_modal_response(self.TRAINING_FINISHED_TITLE)) is not None:
            if close_response == 0:
                # Close
                retval = False
            elif close_response == 1:
                # Keep open
                pass

        return retval

    def _show_close_modal(self, request_close: bool = True) -> bool:
        """If training, show the close modal window to warn the user about closing the GUI.
        Returns False, if the user confirmed to close the GUI (or no training is in progress), True otherwise.
        """
        if request_close:
            # If a crash message has been shown, training isn't running anymore anyway, so just close
            if self._shown_crash_message:
                return False
            # If the modal is already open, close the GUI (but don't terminate training)
            if not GlobalState().shared.is_training or self.window_manager.is_modal_open(self.CLOSE_MODAL_TITLE):
                return False

            close_modal = ModalWindow(
                title=self.CLOSE_MODAL_TITLE,
                message='Training is currently in progress. Are you sure you want to close '
                        'the viewer? The viewer cannot be reopened on this training process!',
                options=[
                    'Keep GUI Open and Continue Training',
                    'Close GUI, but Continue Training',
                    'Close GUI and Terminate Training',
                ],
                option_colors=[
                    'DEFAULT',
                    'WARNING',
                    'ERROR',
                ],
                vertical_options=True,
                close_action=0,
            )
            self.window_manager.add_modal_window(close_modal)

        # If the close modal is open, handle its response
        if (close_response := self.window_manager.get_modal_response(self.CLOSE_MODAL_TITLE)) is not None:
            if close_response == 0:
                # Keep the GUI open and continue training
                pass
            elif close_response == 1:
                # Close the GUI, but continue training
                return False
            elif close_response == 2:
                # Close the GUI and terminate training
                self._shared_state.terminate_training = True
                return False

        return True

    def _show_crash_modal(self) -> bool:
        """If the parent process has crashed, show a modal window to inform the user."""
        retval = True

        # Detect crash of parent process
        if not self._shown_crash_message and (
                self._shared_state.has_renderer_exited
                or not multiprocessing.parent_process().is_alive()
        ):
            self._shown_crash_message = True
            if self._shared_state.is_training:
                self._crash_modal_title = f'{fa.ICON_FA_HEART_CRACK} Training Has Crashed'
                message = 'The training process has crashed. For more information, please see the console logs.'
            else:
                self._crash_modal_title = f'{fa.ICON_FA_HEART_CRACK} Renderer Has Crashed'
                message = ('The renderer has crashed. For more information, please see the console logs. You need '
                           'to restart the application to continue using the GUI.')

            crash_modal = ModalWindow(
                title=self._crash_modal_title,
                message=message,
                options=[
                    'Exit',
                    'Keep Open',
                ],
                option_colors=[
                    'DEFAULT',
                    'DEFAULT',
                ],
                close_action=1,
            )
            self.window_manager.add_modal_window(crash_modal)
            self._backend_window.title += ' (Crashed!)'

        # If the crash modal is open, handle its response
        if (close_response := self.window_manager.get_modal_response(self._crash_modal_title)) is not None:
            if close_response == 0:
                # Close
                retval = False
            elif close_response == 1:
                # Keep open
                pass

        return retval

    def resize(self, width, height):
        """Resizes the backend window and updates the GUI context."""
        GlobalState().resize(width, height)
        self._viewer.resize((width, height))

    def run(self):
        """Runs the GUI loop as long as the backend reports that the window is open."""
        with (self._backend_window as backend):
            self._viewer.initialize()
            running = True
            while running:
                dt = imgui.get_io().delta_time

                # Handle window closing, during training, show a modal window to warn the user
                window_open = backend.prepare_frame()
                running = self._show_close_modal(request_close=not window_open)
                running = running and self._show_training_done_modal()
                running = running and self._show_crash_modal()

                # Handle user inputs
                self._input_handler.handle_inputs(dt)

                # Communicate with the renderer process
                self._send_frame_request(dt)
                self._receive_frame()

                # Update in-memory references to the current GUI state
                self._render()

                # Render the frame
                self._viewer.render()
                if not GlobalState().launch_config.synchronize_extras:
                    self._viewer.render_extras(OverlayState().enabled, CameraState().current_view)
                else:
                    self._viewer.render_extras(OverlayState().enabled)

                # Process screenshots if requested
                self._screenshot()

                # Finish the frame and show it on screen
                backend.finalize_frame()
