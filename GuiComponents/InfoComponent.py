# -- coding: utf-8 --

"""GuiComponents/InfoComponent.py: Informational component for the GUI, showing FPS and hotkeys."""
import math
from itertools import chain

import imgui
from sdl2 import SDL_SCANCODE_F1, SDL_SCANCODE_H

from ICGui.GuiComponents.Base import GuiComponent
from ICGui.GuiConfig import NeRFConfig
import Framework


# pylint: disable=too-few-public-methods
class InfoComponent(GuiComponent):
    """Informational component for the GUI, showing FPS and hotkeys."""
    _HOTKEYS = [SDL_SCANCODE_F1, SDL_SCANCODE_H]
    _TABLE_FLAGS = imgui.TABLE_ROW_BACKGROUND \
                 | imgui.TABLE_BORDERS_OUTER \
                 | imgui.TABLE_SCROLL_X \
                 | imgui.TABLE_SCROLL_Y

    def __init__(self, config: NeRFConfig):
        super().__init__('Info', InfoComponent._HOTKEYS, config=config, closable=True, default_open=True)

    # pylint: disable=arguments-differ
    def _render(self, model_framerate: float = math.nan, framerate_std: float = math.nan, training_iter: int = None):
        """Renders the GUI component."""
        imgui.text(f'FPS (GUI): {imgui.get_io().framerate:.2f}')
        imgui.text(f'FPS (Model): {model_framerate:.2f} ± {framerate_std:.2f}')

        if training_iter is not None:
            imgui.text(f'Training Iteration: {training_iter} / {Framework.config.TRAINING.NUM_ITERATIONS} '
                       f'({training_iter / Framework.config.TRAINING.NUM_ITERATIONS * 100:.1f}%)')
        else:
            imgui.text('Not Training')

        imgui.spacing()

        imgui.text('Hotkeys:')

        with imgui.begin_table('hotkey_table', 2, flags=InfoComponent._TABLE_FLAGS) as t_open:
            if t_open.opened:
                for name, hotkeys in chain(
                    GuiComponent.registered_window_hotkeys.items(),
                    GuiComponent.registered_external_hotkeys.items(),
                    GuiComponent.registered_hotkeys.items(),
                ):
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    imgui.text(name)
                    imgui.table_set_column_index(1)
                    imgui.text(', '.join(hotkeys))
