"""Create Apple Health-style sleep-stage timelines from the processed dataset."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd


STAGES = {
    0: ("Awake", "#F59E0B"),
    4: ("REM", "#22D3EE"),
    1: ("N1", "#93C5FD"),
    2: ("N2", "#3B82F6"),
    3: ("N3 (Deep)", "#4338CA"),
}
STAGE_Y = {stage: index for index, stage in enumerate(STAGES)}


def svg_text(x: float, y: float, value: str, **attributes: object) -> str:
    attrs = " ".join(
        f'{("class" if key == "class_" else key.replace("_", "-"))}="{value}"'
        for key, value in attributes.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{escape(value)}</text>'


def create_plot(data: pd.DataFrame, subjects: list[str], night: str, output: Path) -> None:
    width = 1400
    left, right = 150, 55
    top, bottom = 100, 90
    panel_height, panel_gap = 245, 45
    height = top + bottom + len(subjects) * panel_height + (len(subjects) - 1) * panel_gap
    plot_width = width - left - right

    selected = data[
        data["subject"].isin(subjects)
        & (data["night"].astype(str) == str(night))
        & data["sleep_stage"].isin(STAGES)
    ].copy()
    if selected.empty:
        raise ValueError(f"No scored data found for night {night} and subjects {subjects}")

    max_hours = max((group["epoch"].max() + 1) / 120 for _, group in selected.groupby("subject"))
    max_hours = int(max_hours) + (max_hours % 1 > 0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#0F172A}.axis{font-size:16px;fill:#475569}.stage{font-size:15px;fill:#334155}.subject{font-size:21px;font-weight:600}.title{font-size:28px;font-weight:700}.subtitle{font-size:16px;fill:#64748B}</style>',
        svg_text(left, 42, "Night 1 sleep-stage timelines", class_="title"),
        svg_text(left, 70, "Expert-labelled stage for every 30-second epoch", class_="subtitle"),
    ]

    stage_spacing = 42
    for panel_index, subject in enumerate(subjects):
        subject_data = selected[selected["subject"] == subject].sort_values("epoch")
        if subject_data.empty:
            raise ValueError(f"No scored data found for {subject}, night {night}")

        panel_top = top + panel_index * (panel_height + panel_gap)
        chart_top = panel_top + 40
        chart_bottom = chart_top + (len(STAGES) - 1) * stage_spacing
        parts.append(svg_text(left, panel_top + 12, subject, class_="subject"))

        for stage, (label, _) in STAGES.items():
            y = chart_top + STAGE_Y[stage] * stage_spacing
            parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#CBD5E1" stroke-width="1"/>')
            parts.append(svg_text(left - 14, y + 5, label, class_="stage", text_anchor="end"))

        for hour in range(max_hours + 1):
            x = left + hour / max_hours * plot_width
            parts.append(f'<line x1="{x:.1f}" y1="{chart_top-10}" x2="{x:.1f}" y2="{chart_bottom+10}" stroke="#E2E8F0" stroke-width="1"/>')
            if panel_index == len(subjects) - 1:
                parts.append(svg_text(x, chart_bottom + 38, str(hour), class_="axis", text_anchor="middle"))

        epochs = subject_data["epoch"].astype(int).tolist()
        stages = subject_data["sleep_stage"].astype(int).tolist()
        segment_start = 0
        for index in range(1, len(stages) + 1):
            end_segment = index == len(stages) or stages[index] != stages[segment_start]
            if not end_segment:
                continue
            stage = stages[segment_start]
            _, colour = STAGES[stage]
            start_hour = epochs[segment_start] / 120
            end_hour = (epochs[index - 1] + 1) / 120
            x1 = left + start_hour / max_hours * plot_width
            x2 = left + end_hour / max_hours * plot_width
            y = chart_top + STAGE_Y[stage] * stage_spacing
            parts.append(f'<line x1="{x1:.2f}" y1="{y}" x2="{x2:.2f}" y2="{y}" stroke="{colour}" stroke-width="8" stroke-linecap="butt"/>')

            if index < len(stages):
                next_stage = stages[index]
                next_y = chart_top + STAGE_Y[next_stage] * stage_spacing
                parts.append(f'<line x1="{x2:.2f}" y1="{y}" x2="{x2:.2f}" y2="{next_y}" stroke="#94A3B8" stroke-width="2" opacity="0.65"/>')
            segment_start = index

    final_chart_bottom = top + (len(subjects) - 1) * (panel_height + panel_gap) + 40 + (len(STAGES) - 1) * stage_spacing
    parts.append(svg_text(left + plot_width / 2, final_chart_bottom + 72, "Hours from the start of the recording", class_="axis", text_anchor="middle"))
    parts.append("</svg>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("processed_sleep_data_enhanced_hr.csv"))
    parser.add_argument("--subjects", nargs=3, default=["Bidslab00", "Bidslab01", "Bidslab02"])
    parser.add_argument("--night", default="1")
    parser.add_argument("--output", type=Path, default=Path("sleep_stage_timeline_night1_three_subjects.svg"))
    args = parser.parse_args()

    data = pd.read_csv(args.input, usecols=["subject", "night", "epoch", "sleep_stage"])
    create_plot(data, args.subjects, args.night, args.output)
    print(f"Saved timeline to {args.output.resolve()}")


if __name__ == "__main__":
    main()
