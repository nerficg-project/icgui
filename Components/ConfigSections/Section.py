"""Components/ConfigSections/Section.py: Base class for all configuration sections, rendered as a collapsible header."""

from dataclasses import dataclass, field

from imgui_bundle import imgui

from ICGui.State import PersistentState


@dataclass
class Section:
    name: str = 'Section'
    always_open: bool = False
    default_open: bool = False
    flags: imgui.TreeNodeFlags_ = 0
    _hidden: bool = field(init=False)

    def __post_init__(self):
        self._hidden = PersistentState().section_hidden.get(self.name, False)

    def render(self):
        """Render the section header and its content if open."""
        if self.always_open:
            imgui.separator_text(self.name)
            self._render()
            imgui.spacing()
            return

        flags = self.flags
        if self.default_open or PersistentState().section_expanded.get(self.name, False):
            flags |= imgui.TreeNodeFlags_.default_open
        is_open, keep_open = imgui.collapsing_header(f'{self.name}##header', not self._hidden, flags)
        if is_open:
            PersistentState().section_expanded[self.name] = True
            self._render()
            imgui.spacing()
        else:
            PersistentState().section_expanded[self.name] = False

        if not keep_open:
            self.hidden = True

    def _render(self):
        pass

    @property
    def hidden(self) -> bool:
        return self._hidden

    @hidden.setter
    def hidden(self, value: bool):
        self._hidden = value
        PersistentState().section_hidden[self.name] = value
