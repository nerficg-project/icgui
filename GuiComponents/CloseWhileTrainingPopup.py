# -- coding: utf-8 --

"""GuiComponents/CloseWhileTrainingPopup.py: Popup to ask for confirmation when closing during training."""
from enum import Enum

import imgui

from ICGui.GuiComponents.Base import GuiComponent


# pylint: disable=too-few-public-methods
class CloseWhileTrainingPopup(GuiComponent):
    """Informational component for the GUI, showing FPS and hotkeys."""
    class Response(Enum):
        """Response to the popup."""
        KEEP_OPEN = 0
        TERMINATE = 1
        CLOSE = 2

    _WINDOW_FLAGS = imgui.WINDOW_NO_SAVED_SETTINGS \
                  | imgui.WINDOW_NO_COLLAPSE \
                  | imgui.WINDOW_NO_RESIZE

    def __init__(self):
        super().__init__('Close Window?', [],
                         closable=True,
                         default_open=False,
                         force_size=(450, 180),
                         force_center=True,
                         force_focus=True,
                         window_flags=CloseWhileTrainingPopup._WINDOW_FLAGS)
        self.close()
        self.response: CloseWhileTrainingPopup.Response = CloseWhileTrainingPopup.Response.KEEP_OPEN

    def open(self):
        """Opens the popup and resets the user response."""
        super().open()
        self.response = None

    # pylint: disable=arguments-differ
    def _render(self):
        """Renders the GUI component."""
        imgui.text_wrapped('Training is currently running. Are you sure you want to close the GUI?')

        imgui.spacing()

        terminate_training = imgui.button('Terminate Training and GUI')
        close_gui = imgui.button('Close GUI, but continue Training')
        keep_open = imgui.button('Keep GUI open and continue Training')

        if keep_open:
            self.response = CloseWhileTrainingPopup.Response.KEEP_OPEN
            self.close()

        if close_gui:
            self.response = CloseWhileTrainingPopup.Response.CLOSE

        if terminate_training:
            self.response = CloseWhileTrainingPopup.Response.TERMINATE
