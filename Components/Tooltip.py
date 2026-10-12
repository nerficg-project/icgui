from imgui_bundle import imgui


def tooltip(text: str, text_wrap_em: float = 35.0, flags: int = 0):
    if imgui.is_item_hovered(flags):
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * text_wrap_em)
        imgui.text_wrapped(text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()
