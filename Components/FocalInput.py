"""Components/FocalInput.py: Input element for focal length, allowing both pixel and fov degree input."""

from imgui_bundle import imgui, imgui_ctx

from Cameras.utils import focal_to_fov, fov_to_focal


_MIN_DEG, _MAX_DEG = 1e-6, 170.0
_MIN_F_NORM, _MAX_F_NORM = fov_to_focal(_MAX_DEG, degrees=True), fov_to_focal(_MIN_DEG, degrees=True)
_INPUT_SPEED = 1.0 / 500.0
_SPEED_DEG = (_MAX_DEG - _MIN_DEG) * _INPUT_SPEED


def focal_input(label: str, val: float, size: float) -> tuple[bool, float]:
    """Renders a drag input for focal length, allowing for both pixel and degree input.
    Args:
        label (str): The label for the input field.
        val (float): The current focal length in pixels.
        size (float): The width / height of the camera in px.
    """
    any_changed = False
    new_val = val

    # Calculate input speed, min, and max for the focal length input in pixels
    speed_f = _INPUT_SPEED * size
    min_f = _MIN_F_NORM * size
    max_f = _MAX_F_NORM * size

    # Set item width to half of the available width, minus spacing
    item_width = imgui.calc_item_width()
    spacing = imgui.get_style().item_inner_spacing.x
    with imgui_ctx.push_item_width(item_width * 0.5):
        # Focal length input in pixels
        changed, new_px = imgui.drag_float(
            f'##{label}_px', val,
            speed_f, min_f, max_f,
            '%.3f px',
        )
        if changed:
            any_changed = True
            new_val = max(min_f, min(max_f, new_px))

        imgui.same_line(spacing=spacing)

        # Focal length input in degrees
        changed, new_deg = imgui.drag_float(
            label, focal_to_fov(val / size, degrees=True),
            _SPEED_DEG, _MIN_DEG, _MAX_DEG,
            '%.1f°',
        )
        if changed:
            any_changed = True
            # Convert to pixel focal length
            new_val = fov_to_focal(max(_MIN_DEG, min(_MAX_DEG, new_deg)), degrees=True) * size

    return any_changed, new_val