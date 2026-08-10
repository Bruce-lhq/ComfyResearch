"""moving_statistics ? FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="moving_statistics",
        label="Moving statistics",
        category="analysis",
        hint="Rolling-window mean or standard deviation over a rank-1 (1D) tensor with configurable window, stride, and min periods; pass the result downstream as tensor.",
        fields=(
            EnumField(key="statistic", label="Statistic", default="mean"),
            IntField(key="window", label="Window", default=10, min=1, max=1024),
            IntField(key="stride", label="Stride", default=1, min=1, max=1024),
            IntField(key="minPeriods", label="Min Periods", default=1, min=1, max=1024),
            BoolField(key="logScaleX", label="Log Scale X", default=False),
            BoolField(key="logScaleY", label="Log Scale Y", default=False),
        ),
        defaults=(
            ("statistic", "mean"),
            ("window", 10),
            ("stride", 1),
            ("minPeriods", 1),
            ("logScaleX", False),
            ("logScaleY", False),
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="MovingStatisticsNode"),
    )
)
