"""Visualization helpers for tetrode data.

Holds the Open Ephys "Classic" channel palette (the 16-color
``DefaultColourScheme`` from the Open Ephys GUI LFP Viewer plugin) and a helper
to map tetrode labels to those colors — used to color loupe dense traces and
spike rasters consistently by tetrode.

Reference: open-ephys/plugin-GUI, ``Plugins/LfpViewer/ColourSchemes/
DefaultColourScheme.cpp``. It has exactly 16 channel colors — one per tetrode.
"""

from __future__ import annotations

import re

# 16 RGB channel colors from Open Ephys DefaultColourScheme, in order.
OPEN_EPHYS_CLASSIC_COLORS: list[tuple[int, int, int]] = [
    (224, 185, 36),
    (214, 210, 182),
    (243, 119, 33),
    (186, 157, 168),
    (237, 37, 36),
    (179, 122, 79),
    (217, 46, 171),
    (217, 139, 196),
    (101, 31, 255),
    (141, 111, 181),
    (48, 117, 255),
    (184, 198, 224),
    (116, 227, 156),
    (150, 158, 155),
    (82, 173, 0),
    (125, 99, 32),
]

# Dark navy background used by the same scheme.
OPEN_EPHYS_CLASSIC_BACKGROUND: tuple[int, int, int] = (0, 18, 43)


def _tetrode_sort_key(label) -> tuple[int, object]:
    """Sort ``TT<k>`` labels by their integer ``k``; others lexicographically."""
    m = re.fullmatch(r"TT(\d+)", str(label))
    return (0, int(m.group(1))) if m else (1, str(label))


def tetrode_color_map(
    tetrode_labels,
    colors: list[tuple[int, int, int]] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Map each tetrode label to an Open Ephys Classic color.

    Labels are de-duplicated and ordered by tetrode number (``TT1``, ``TT2``,
    …), so ``TT1`` always gets the first color, etc. Colors cycle if there are
    more tetrodes than palette entries (>16).
    """
    palette = colors if colors is not None else OPEN_EPHYS_CLASSIC_COLORS
    labels = sorted(dict.fromkeys(tetrode_labels), key=_tetrode_sort_key)
    return {str(lab): palette[i % len(palette)] for i, lab in enumerate(labels)}
