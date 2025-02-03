# -- coding: utf-8 --

"""GuiComponents/TrainingDonePopup.py: Popup to notify user that training has finished."""
import sys

import imgui

from ICGui.GuiComponents.Base import GuiComponent


# pylint: disable=too-few-public-methods
class TrainingDonePopup(GuiComponent):
    """Informational component for the GUI, showing FPS and hotkeys."""
    _WINDOW_FLAGS = imgui.WINDOW_NO_SAVED_SETTINGS \
                  | imgui.WINDOW_NO_COLLAPSE \
                  | imgui.WINDOW_NO_RESIZE

    def __init__(self):
        super().__init__('Training Finished', [],
                         closable=True,
                         default_open=False,
                         force_size=(400, 140),
                         force_center=True,
                         force_focus=True,
                         window_flags=TrainingDonePopup._WINDOW_FLAGS)
        self.close()

    # pylint: disable=arguments-differ
    def _render(self):
        """Renders the GUI component."""
        imgui.text_wrapped('Training has successfully finished. Close the GUI or keep it open?')

        imgui.spacing()

        close_gui = imgui.button('Close')
        imgui.same_line()
        keep_open = imgui.button('Keep Open')

        if keep_open:
            self.close()
            return

        if close_gui:
            sys.exit(0)
