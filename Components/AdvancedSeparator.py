"""Components/AdvancedSeparator.py: Provides a separator component with a label and button."""

from imgui_bundle import imgui, imgui_ctx

from ICGui.Components.ButtonBehavior import button_behavior


def advanced_separator(label_text: str, button_text: str):
    text_size = imgui.calc_text_size(label_text)
    button_size = imgui.calc_text_size(button_text)
    text_padding = imgui.get_style().separator_text_padding.x
    with imgui_ctx.begin_table(f'{label_text}##separator_table', 5):
        imgui.push_id(label_text)
        imgui.table_setup_column('SepStart', imgui.TableColumnFlags_.width_fixed, text_padding)
        imgui.table_setup_column('LabelText', imgui.TableColumnFlags_.width_fixed, text_size.x)
        imgui.table_setup_column('SepMiddle', imgui.TableColumnFlags_.width_stretch, 100)
        imgui.table_setup_column('Button', imgui.TableColumnFlags_.width_fixed, button_size.x)
        imgui.table_setup_column('SepEnd', imgui.TableColumnFlags_.width_fixed, text_padding)
        imgui.table_next_row()

        center_y = imgui.get_cursor_pos_y() + max(imgui.get_frame_height(), imgui.calc_text_size(label_text).y) / 2

        imgui.table_set_column_index(0)
        imgui.set_cursor_pos_y(center_y - imgui.get_style().separator_text_border_size / 2)
        imgui.separator()
        imgui.table_set_column_index(1)
        imgui.set_cursor_pos_y(center_y - text_size.y / 2)
        imgui.text(label_text)
        imgui.table_set_column_index(2)
        imgui.set_cursor_pos_y(center_y - imgui.get_style().separator_text_border_size / 2)
        imgui.separator()
        imgui.table_set_column_index(3)
        imgui.set_cursor_pos_y(center_y - button_size.y / 2)
        with button_behavior(f'{button_text}##{label_text}') as result:
            imgui.text(button_text)
        imgui.table_set_column_index(4)
        imgui.set_cursor_pos_y(center_y - imgui.get_style().separator_text_border_size / 2)
        imgui.separator()
        imgui.pop_id()
    return result.pressed
