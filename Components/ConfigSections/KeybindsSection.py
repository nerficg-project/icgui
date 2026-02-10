"""Components/ConfigSections/KeybindsSection.py: Keybind configuration section for the config window."""

from dataclasses import dataclass

from imgui_bundle import imgui, imgui_ctx, hello_imgui, icons_fontawesome_6 as fa

from ICGui.Components.HelpIndicator import help_indicator
from ICGui.State.Volatile import GlobalState
from ICGui.util.Enums import Modifier
from .Section import Section


@dataclass
class KeybindsSection(Section):
    name: str = f'{fa.ICON_FA_KEYBOARD} Keybinds'
    always_open: bool = False
    default_open: bool = False

    _TABLE_FLAGS: imgui.TableFlags_ = imgui.TableFlags_.row_bg \
                                      | imgui.TableFlags_.borders_outer \
                                      | imgui.TableFlags_.scroll_x \
                                      | imgui.TableFlags_.scroll_y

    def _render(self):
        help_indicator('Keybinds are only shown here for reference and cannot be changed yet.')

        with imgui_ctx.begin_table(f'{self.name}_table', 2, flags=self._TABLE_FLAGS,
                                   outer_size=hello_imgui.em_to_vec2(0, 12)) as table_open:
            if not table_open:
                return
            for action, (description, hotkeys) in GlobalState().input_manager.hotkeys.items():
                imgui.push_id(description)
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(description)
                imgui.table_set_column_index(1)
                for i, (modifier, key, is_editable) in enumerate(hotkeys):
                    # TODO: Make hotkeys editable
                    imgui.push_id(str(i))
                    show_key = imgui.button if is_editable else imgui.text

                    # Each keybind button gets a different color
                    hue = i * 0.15 % 1.0  # Cycle through hues for each keybind
                    imgui.push_style_color(imgui.Col_.button, imgui.ImColor.hsv(hue, 0.5, 0.4).value)
                    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImColor.hsv(hue, 0.6, 0.5).value)
                    imgui.push_style_color(imgui.Col_.button_active, imgui.ImColor.hsv(hue, 0.7, 0.6).value)

                    key_name = key.title() if isinstance(key, str) else imgui.get_key_name(key)
                    if modifier != Modifier.NONE:
                        key_text = f'{modifier.name.title()} + {key_name}'
                    else:
                        key_text = key_name
                    show_key(key_text)

                    imgui.pop_style_color(3)
                    imgui.same_line()
                    imgui.pop_id()
                imgui.pop_id()