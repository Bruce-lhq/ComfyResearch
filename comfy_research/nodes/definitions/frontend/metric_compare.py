"""metric_compare — frontend-only comparison of two resolved curve visualizations."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FrontendNodeDef, FrontendSpec, InPort, PortAccept

# 1D-series-capable viz sources: curve series viz, observable mirrors
# (observable_viz / observable_accuracy), training viz, and the tensor viz
# display family (tensor_viz_0d/1d/2d/general). Sources that don't resolve to a
# comparable 1D series surface a warning on the Metric compare node instead of
# failing at the port level, so any 1D tensor visualization can be wired here.
_COMPARE_SOURCES = (
    PortAccept(handles=("compare",), source_type="curve_series_viz"),
    PortAccept(handles=("compare",), source_type="observable_viz"),
    PortAccept(handles=("compare",), source_type="observable_accuracy"),
    PortAccept(handles=("compare",), source_type="training_visualization"),
    PortAccept(handles=("compare",), source_family="observable_user_tensor_viz_display"),
)


DEF = frontend_node_def(
    FrontendNodeDef(
        type="metric_compare",
        label="Metric compare",
        category="visualization",
        fields=(
            EnumField(key="layout", label="Layout", default="horizontal", options=("horizontal", "vertical", "overlay")),
        ),
        defaults=(
            ("layout", "horizontal"),
            ("logScaleX", False),
            ("logScaleY", False),
        ),
        ports=(
            InPort(id="left", accepts=_COMPARE_SOURCES),
            InPort(id="right", accepts=_COMPARE_SOURCES),
        ),
        frontend=FrontendSpec(component_key="MetricCompareNode"),
    )
)
