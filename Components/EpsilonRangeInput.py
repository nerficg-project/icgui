"""Components/EpsilonRangeInput.py: Range input which requires the values to be at least epsilon apart at all times."""

from typing import Sequence

from imgui_bundle import imgui, imgui_ctx


def epsilon_range_input(label: str, vals: Sequence[float], speed: float, min_v: float, max_v: float,
                        epsilon: float = 1e-6, fmt='%.3f') -> tuple[bool, Sequence[float]]:
    """Renders a two drag inputs representing a range, ensuring the two values are never < epsilon apart.
    Args:
        label (str): The label for the input field.
        vals (Sequence[float]): The current values for the range, should be of length 2.
        speed (float): The speed at which the values change when dragging.
        min_v (float): The minimum value for the range.
        max_v (float): The maximum value for the range.
        epsilon (float): The minimum difference between the two values.
        fmt (str): The format string for displaying the values.
    """
    any_changed = False
    new_vals = vals

    # Set item width to half of the available width, minus spacing
    item_width = imgui.calc_item_width()
    spacing = imgui.get_style().item_inner_spacing.x
    with imgui_ctx.push_item_width(item_width * 0.5):
        # Lower range input
        changed_lower, new_lower = imgui.drag_float(
            f'##{label}_lower', vals[0],
            v_speed=speed, v_min=min_v, v_max=max_v - epsilon, format=fmt,
        )
        if changed_lower:
            any_changed = True
            new_vals = (
                max(min_v, min(max_v - epsilon, new_lower)),
                max(new_lower + epsilon, vals[1]),
            )

        imgui.same_line(spacing=spacing)

        # Upper range input
        changed_upper, new_upper = imgui.drag_float(
            f'{label}##upper', vals[1],
            v_speed=speed, v_min=min_v + epsilon, v_max=max_v, format=fmt,
        )
        if changed_upper:
            any_changed = True
            new_vals = (
                min(new_upper - epsilon, vals[0]),
                max(min_v + epsilon, new_upper),
            )

    return any_changed, new_vals
