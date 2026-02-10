from imgui_bundle import imgui


def help_indicator(help_text: str, text_wrap_em: float = 35.0):
    """Display a little (?) mark which shows a tooltip when hovered."""
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * text_wrap_em)
        imgui.text_wrapped(help_text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
