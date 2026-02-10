"""Components/Windows/ConfigWindow.py: Window containing GUI configuration sections."""

from dataclasses import dataclass, field
from typing import ClassVar

from imgui_bundle import imgui, icons_fontawesome_6 as fa

import Framework
from ICGui.Backend import FontManager
from ICGui.Components.ButtonBehavior import button_behavior
from ICGui.Components.ConfigSections import (Section,
    AdvancedSection, CameraSection, ControlsSection, DynamicSection,
    KeybindsSection, OutputSelectionSection, OverlaysSection,
    PerformanceSection, PoseSection, ResolutionSection, ScreenshotSection
)
from ICGui.Components.LinkButton import link_button, link_text
from ICGui.State.Volatile import GlobalState
from .Window import Window


@dataclass
class ConfigWindow(Window):
    """Window containing GUI configuration sections."""
    name: ClassVar[str] = f'Configuration'
    default_open: ClassVar[bool] = True
    sections: list[Section] = field(default_factory=list)

    _absolute_paths: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.sections.append(PerformanceSection())
        self.sections.append(PoseSection())
        self.sections.append(OutputSelectionSection())
        try:
            self.sections.append(DynamicSection())
        except DynamicSection.DynamicError:
            # If the dataset does not contain multiple timesteps, we skip the dynamic section
            pass
        self.sections.append(ResolutionSection())
        self.sections.append(CameraSection())
        self.sections.append(ScreenshotSection())
        self.sections.append(OverlaysSection())
        self.sections.append(AdvancedSection())
        self.sections.append(ControlsSection())
        self.sections.append(KeybindsSection())

    def _render(self, **kwargs):
        """Renders the GUI component."""
        # Render current model info
        self._render_model_info()

        imgui.spacing()

        for section in self.sections:
            if section.hidden:
                continue
            section.render()

        imgui.spacing()
        imgui.spacing()

        self._render_section_dropdown()

        imgui.separator()
        imgui.spacing()

        self._render_footer()

    def _render_model_info(self):
        FontManager.bold(imgui.text)(f'Model:')
        imgui.same_line()
        FontManager.muted_color(imgui.text)(Framework.config.TRAINING.MODEL_NAME)

        config_path = GlobalState().launch_config.training_config_path
        if config_path is not None:
            with button_behavior('Config') as button:
                FontManager.bold(imgui.text)(f'Config:')
                imgui.same_line()
                text = str(config_path.absolute() if self._absolute_paths else config_path)
                FontManager.muted_color(imgui.text)(text)
            self._absolute_paths ^= button.pressed

        checkpoint_path = GlobalState().launch_config.checkpoint_path
        if checkpoint_path is not None:
            with button_behavior('Checkpoint') as button:
                FontManager.bold(imgui.text)(f'Checkpoint:')
                imgui.same_line()
                text = str(checkpoint_path.absolute() if self._absolute_paths else checkpoint_path)
                FontManager.muted_color(imgui.text)(text)
            self._absolute_paths ^= button.pressed

    def _render_section_dropdown(self):
        """Renders a dropdown to re-show hidden sections."""
        hidden_sections = [s for s in self.sections if s.hidden]
        if not hidden_sections:
            return

        if imgui.begin_combo('Add Section', 'Select Section'):
            for section in hidden_sections:
                if imgui.selectable(section.name, False)[0]:
                    section.hidden = False
            imgui.end_combo()

    @staticmethod
    def _render_footer():
        """Renders the footer with links and version information."""
        # Align the footer to the bottom of the window
        if imgui.get_content_region_avail().y - imgui.get_frame_height() > 0:
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + imgui.get_content_region_avail().y - imgui.get_frame_height())

        link_text('© 2025 NeRFICG Project', 'https://github.com/nerficg-project/nerficg/blob/main/LICENSE')
        imgui.set_item_tooltip('See the license')

        item_spacing = imgui.get_style().item_inner_spacing.x
        imgui.same_line(spacing=item_spacing)

        # Get size of the resize grip (we want to avoid drawing behind it)
        resize_grip_size = max(imgui.get_font_size() * 1.1, imgui.get_style().window_rounding + 1.0 + imgui.get_font_size() * 0.2)

        # Calculate available space for the links, considering the resize grip and icons
        x_target = imgui.get_content_region_avail().x - resize_grip_size
        x_target -= imgui.calc_text_size(fa.ICON_FA_CODE).x + imgui.calc_text_size(fa.ICON_FA_BUG).x
        x_target -= imgui.get_style().frame_padding.x * 4.0

        # Align to the right side of the window if possible, otherwise just wrap to the next line
        if x_target >= imgui.get_style().item_spacing.x:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + x_target)
        else:
            imgui.new_line()

        # Draw GitHub links
        link_button(fa.ICON_FA_CODE, 'https://github.com/nerficg-project/nerficg')
        imgui.set_item_tooltip('Check out the code')
        imgui.same_line(spacing=item_spacing)
        link_button(fa.ICON_FA_BUG, 'https://github.com/nerficg-project/nerficg/issues/new')
        imgui.set_item_tooltip('Report an issue')
