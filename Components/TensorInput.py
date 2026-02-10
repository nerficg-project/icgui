"""Components/TensorInput.py: Tensor input component with input as either separate float inputs, or one large text input."""

import numpy as np
from imgui_bundle import imgui, imgui_ctx, icons_fontawesome_6 as fa

from ICGui.Backend import FontManager
from ICGui.Components.Colors import apply_ui_color
from ICGui.util.Cameras import parse_mat4


_INPUT_CACHE: dict[str, dict[tuple[int, int], float]] = {}


def input_mat4(label: str, matrix: np.ndarray, as_separate: bool = False) -> tuple[bool, np.ndarray]:
    """Renders a 4x4 matrix input.
    Args:
        label (str): The label for the input field.
        matrix (np.ndarray): The 4x4 numpy array to edit.
        as_separate (bool): If True, renders the matrix as separate float inputs for each entry,
            otherwise renders it as a multi-line text input.
    """
    assert matrix.shape == (4, 4)
    imgui.push_font(FontManager.current.bold, 0.0)
    imgui.text(label)
    imgui.pop_font()

    if as_separate:
        _input_cache = _INPUT_CACHE.setdefault(label, {})

        # Render accept / cancel buttons
        cursor_pos = imgui.get_cursor_pos()
        button_pos = imgui.get_cursor_pos()
        button_pos.x += imgui.calc_item_width() + imgui.get_style().item_spacing.x
        imgui.set_cursor_pos(button_pos)
        item_height = imgui.get_frame_height()
        line_height = imgui.get_frame_height_with_spacing()
        imgui.begin_disabled(len(_input_cache) == 0)
        with apply_ui_color('DEFAULT' if len(_input_cache) == 0 else 'ACCEPT'):
            accepted = imgui.button(
                f'{fa.ICON_FA_CHECK}##{label}',
                (0, line_height + item_height)
            )
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text('Accept changes')
            imgui.end_tooltip()
        button_pos.y += line_height * 2
        imgui.set_cursor_pos(button_pos)
        with apply_ui_color('DEFAULT' if len(_input_cache) == 0 else 'ERROR'):
            cancelled = imgui.button(
                f'{fa.ICON_FA_BAN}##{label}',
                (0, line_height + item_height)
            )
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text('Discard changes')
            imgui.end_tooltip()
        imgui.set_cursor_pos(cursor_pos)
        imgui.end_disabled()

        # If cancel button is pressed, reset the input cache
        if cancelled:
            _input_cache.clear()

        # Calculate width of individual input fields
        item_width = imgui.calc_item_width() * 0.25
        spacing = imgui.get_style().item_inner_spacing.x
        item_width -= spacing * 3 / 4  # Adjust for spacing between inputs

        with imgui_ctx.push_item_width(item_width):
            # Render rows
            for row in range(4):
                # Render columns
                for col in range(4):
                    if (row, col) in _input_cache:
                        style = 'MODIFIED'
                    else:
                        style = 'DEFAULT'

                    with apply_ui_color(style):
                        changed, value = imgui.input_float(
                            f'##{label}_{row}_{col}',
                            _input_cache.get((row, col), matrix[row, col].item()),
                            format='%.3f',
                        )
                    if changed:
                        _input_cache[(row, col)] = value
                    if accepted:
                        matrix[row, col] = value
                        if (row, col) in _input_cache:
                            del _input_cache[(row, col)]

                    imgui.same_line(spacing=spacing)
                imgui.new_line()

        return accepted, matrix
    else:
        changed, mat_text = imgui.input_text_multiline(
            f'##{matrix}', str(matrix), (-1, imgui.get_frame_height() * 4),
            flags=imgui.InputTextFlags_.enter_returns_true
                  | imgui.InputTextFlags_.allow_tab_input
                  | imgui.InputTextFlags_.ctrl_enter_for_new_line,
        )
        if not changed:
            return False, matrix

        # Try to parse the input text as a 4x4 tensor
        try:
            match = parse_mat4(mat_text)
        except ValueError:
            return False, matrix

        return True, match
