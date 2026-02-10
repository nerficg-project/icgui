"""Components/FontInput.py: Font input component with font family selection and size adjustment, as well as a preview
of the selected font."""

from functools import partial

from imgui_bundle import imgui, hello_imgui, imgui_ctx, icons_fontawesome_6 as fa

from ICGui.Backend.FontManager import FontManager, FontSpec
from ICGui.util import AutoComplete


_FILLER_TEXT = [
    'NeRFICG',
    'Moritz Kappel, Florian Hahlbohm, and Timon Scholz',
    'A flexible Pytorch framework for simple and efficient implementation of ' 
    'neural radiance fields and rasterization-based view synthesis methods, '
    'including a GUI for interactive rendering.',
]
_expanded: dict[str, bool] = {}


def font_input(label: str, value: FontSpec):
    """Renders a font input field with a text input for the font family, a dropdown for available fonts,
    and an input field for the font size. Also provides a preview of the selected font."""
    global _expanded

    # Show font family text input
    # Note: Because the callback_completion handler does not respond to Shift+Tab, we use the callback_always flag.
    #       This allows us to check for the correct key / key combination ourselves. To prevent Shift+Tab from
    #       triggering tab navigation, we also need to set the input element as the owner of the tab key.
    flags = (imgui.InputTextFlags_.callback_always  # imgui.InputTextFlags_.callback_completion
             | imgui.InputTextFlags_.callback_edit)
    callback = partial(_font_input_callback, label=label, value=value.name)
    modified, value.name = imgui.input_text(
        f'{label} Family', value.name,
        flags=flags, callback=callback,
    )
    imgui.set_item_key_owner(imgui.Key.tab)

    # Show dropdown arrow button
    _expanded.setdefault(label, False)
    _expanded[label] ^= imgui.button(
        f'{fa.ICON_FA_CARET_DOWN if _expanded[label] else fa.ICON_FA_CARET_RIGHT}##{label}'
    )
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.text('Show preview')
        imgui.end_tooltip()

    imgui.same_line()

    # Show an input field for the font size
    _, value.size = imgui.input_int(f'{label} Size', value.size, 1, 8)

    # Show dropdown list if expanded
    if _expanded[label]:
        height = 10  # Height in em
        with imgui_ctx.begin_list_box(f'##listbox_{label}', hello_imgui.em_to_vec2(0, height)):
            for font in FontManager.AVAILABLE_FONTS:
                selected = font == value.name
                clicked, _ = imgui.selectable(font, selected)

                # Make sure the selected item is focused on first render
                if selected:
                    imgui.set_item_default_focus()

                # If the font was changed using the text input, scroll to the selected item
                if selected and modified:
                    imgui.set_scroll_here_y()

                # Handle font selection via clicking
                if clicked:
                    AutoComplete.invalidate(label)  # Invalidate auto-complete options
                    value.name = font
                    modified = True

        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        with imgui_ctx.begin_child(f'##preview_{label}', hello_imgui.em_to_vec2(0, height), imgui.ChildFlags_.borders):
            # Header
            imgui.push_font(FontManager.current.bold, 0.0)
            imgui.text(f'Preview')
            imgui.pop_font()

            # Render a preview of the selected font
            preview_font = FontManager.load_font(value.name, size=0)
            imgui.push_font(preview_font.bold, value.size)
            imgui.text_wrapped(_FILLER_TEXT[0])
            imgui.pop_font()

            imgui.push_font(preview_font.italics, value.size)
            imgui.text_wrapped(_FILLER_TEXT[1])
            imgui.pop_font()

            imgui.push_font(preview_font.normal, value.size)
            imgui.text_wrapped(_FILLER_TEXT[2])
            imgui.pop_font()

        imgui.spacing()

    return modified, value


def _font_input_callback(data: imgui.InputTextCallbackData, label: str, value: str) -> int:
    """Auto-complete callback function for the font input field."""
    # Invalidate the current auto-complete options on any manual edit
    if data.event_flag & imgui.InputTextFlags_.callback_edit:
        AutoComplete.invalidate(label)
        return 0

    # Handle auto-completion, see note in font_input function
    if imgui.is_key_pressed(imgui.Key.tab):
        AutoComplete.font(label, value, event_data=data)

    return 0
