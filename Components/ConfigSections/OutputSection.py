"""Components/ConfigSections/OutputSection.py: Output and colormap configuration section for the config window."""

import math
from dataclasses import dataclass

from imgui_bundle import imgui, hello_imgui, icons_fontawesome_6 as fa
import torch

from ICGui.Backend import CudaOpenGL
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.EpsilonRangeInput import epsilon_range_input
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.util.ColorChannels import apply_color_map
from ICGui.State.Volatile import ViewerOutputState, ColorMapState
from .Section import Section


@dataclass
class OutputSelectionSection(Section):
    name: str = f'{fa.ICON_FA_BRUSH} Output Selection'
    always_open: bool = False
    default_open: bool = False
    _color_map_previews: imgui.ImTextureRef | None = None

    def __post_init__(self):
        super().__post_init__()

    def _render(self):
        self._render_output_selection()

        if ColorMapState().show_settings:
            imgui.separator_text(f'{fa.ICON_FA_PALETTE} Color Map Configuration')
            self._render_color_map_selection()
            self._render_color_map_configuration()

    @staticmethod
    def _render_output_selection():
        output = ViewerOutputState()
        _, output.selected_idx = imgui.combo(
            'Model Output', output.selected_idx,
            output.output_choices,
        )
        if output.selected_idx >= len(output.output_choices):
            output.selected_idx = 0

    def _render_color_map_selection(self):
        cmap = ColorMapState()
        if self._color_map_previews is None:
            # Draw the color map preview texture once
            self._generate_color_map_preview()

        # Calculate texture grid size
        grid_size_x = math.ceil(math.sqrt(len(ColorMapState().COLORMAP_CHOICES)))
        grid_size_y = math.ceil(len(ColorMapState().COLORMAP_CHOICES) / grid_size_x)
        tile_size = math.floor(hello_imgui.em_size())

        # Calculate preview of current selected
        x = cmap.selected_idx % grid_size_y
        y = cmap.selected_idx // grid_size_y

        # Render the current selected color map preview as an image button (non-interactive)
        imgui.push_style_color(imgui.Col_.button, imgui.get_style_color_vec4(imgui.Col_.frame_bg))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.get_style_color_vec4(imgui.Col_.frame_bg))
        imgui.push_style_color(imgui.Col_.button_active, imgui.get_style_color_vec4(imgui.Col_.frame_bg))
        imgui.image_button('##current_preview', self._color_map_previews, (tile_size, tile_size),
                           (x * tile_size / (grid_size_x * tile_size), y * tile_size / (grid_size_y * tile_size)),
                           ((x + 1) * tile_size / (grid_size_x * tile_size),
                            (y + 1) * tile_size / (grid_size_y * tile_size)))
        imgui.pop_style_color(3)

        # Ensure the combo box is aligned with the image button
        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
        imgui.set_next_item_width(imgui.calc_item_width() - (tile_size + 2 * imgui.get_style().frame_padding.x
                                                             + imgui.get_style().item_inner_spacing.x))

        # Render the combo box with all color map options, with the corresponding preview tile
        if imgui.begin_combo('Color Map', cmap.COLORMAP_CHOICES[cmap.selected_idx].title()):
            for idx, color_map in enumerate(cmap.COLORMAP_CHOICES):
                x = idx % grid_size_y
                y = idx // grid_size_y
                imgui.image(self._color_map_previews, (tile_size, tile_size),
                            (x * tile_size / (grid_size_x * tile_size), y * tile_size / (grid_size_y * tile_size)),
                            ((x + 1) * tile_size / (grid_size_x * tile_size), (y + 1) * tile_size / (grid_size_y * tile_size)))
                imgui.same_line()
                if imgui.selectable(color_map.title(), idx == cmap.selected_idx)[0]:
                    cmap.selected_idx = idx
            imgui.end_combo()
        help_indicator('The color map to apply to monochromatic outputs (e.g. depth, alpha etc.)')

    @staticmethod
    def _render_color_map_configuration():
        cmap = ColorMapState()

        # nth-percentile for automatic min/max calculation
        _, cmap.nth_percentile = imgui.slider_float(
            'n-th Percentile', cmap.nth_percentile,
            0.1, 100.0, format='%.2f%%')
        help_indicator('The percentage of pixels to use for automatically determining '
                       'the minimum / maximum to use for the selected color map.')

        # Custom Min/Max Range
        if cmap.custom_min_max:
            changed, new_min_max = epsilon_range_input(
                'Min/Max', cmap.min_max,
                speed=cmap.min_max_change_speed,
                min_v=-10_000 if not cmap.logscale else -1 + 1e-6,
                max_v=10_000,
                epsilon=1e-6,
            )

            if changed:
                cmap.min_max = tuple[float, float](new_min_max)

            if imgui.button('Reset Min/Max to nth-percentile'):
                cmap.reset_bounds_on_next_update = True

        changed, cmap.custom_min_max = styled_toggle('Custom Min/Max', cmap.custom_min_max)
        if changed:
            cmap.reset_bounds_on_next_update = True

        # Misc settings
        _, cmap.interpolate = styled_toggle('Interpolate', cmap.interpolate)
        help_indicator('Smooth color map gradients by interpolating between discrete color map values.')
        _, cmap.invert = styled_toggle('Invert', cmap.invert)
        help_indicator('Inverts the color map, i.e., low values become high values and vice versa.')
        _, cmap.logscale = styled_toggle('Log-Scale', cmap.logscale)
        help_indicator('Applies logarithmic scaling to the input values before mapping them to colors. ')

    def _generate_color_map_preview(self):
        # Draw texture grid containing all color map previews
        grid_size_x = math.ceil(math.sqrt(len(ColorMapState().COLORMAP_CHOICES)))
        grid_size_y = math.ceil(len(ColorMapState().COLORMAP_CHOICES) / grid_size_x)
        tile_size = math.floor(hello_imgui.em_size())
        texture = torch.ones((4, grid_size_y * tile_size, grid_size_x * tile_size), device='cuda', dtype=torch.float32)

        for x in range(grid_size_x):
            for y in range(grid_size_y):
                if x * grid_size_y + y >= len(ColorMapState().COLORMAP_CHOICES):
                    break
                idx = x * grid_size_y + y
                color_map = ColorMapState().COLORMAP_CHOICES[idx]

                gradient = torch.linspace(1, 0, steps=tile_size, device='cuda')
                gradient = gradient.unsqueeze(0).repeat(tile_size, 1).unsqueeze(0)

                texture[:3, y * tile_size:(y+1) * tile_size, x * tile_size:(x+1) * tile_size] = apply_color_map(
                    color_map, gradient, logscale=False, min_max=(0.0, 1.0),
                    interpolate=False, invert=False
                )

        # Generate the OpenGL texture
        texture_id = CudaOpenGL.generate_texture(texture.shape[1:])

        # Map the texture to CUDA and copy the tensor data
        cuda_texture = CudaOpenGL.register_cuda_texture(texture_id)
        CudaOpenGL.draw_tensor_to_texture(texture.permute(2,1,0).contiguous(), cuda_texture)
        CudaOpenGL.unregister_cuda_texture(cuda_texture)

        # Create an ImGui texture from the OpenGL texture
        self._color_map_previews = imgui.ImTextureRef(texture_id)
