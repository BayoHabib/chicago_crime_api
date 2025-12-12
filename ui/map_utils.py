"""Map visualization utilities.

Shared map logic that can be used by Streamlit now
and adapted for React/Leaflet later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import folium
from folium.plugins import HeatMap

if TYPE_CHECKING:
    from .api_client import HotspotResult, PredictionResult

# Chicago bounds
CHICAGO_CENTER = (41.8781, -87.6298)
CHICAGO_BOUNDS = {
    "lat_min": 41.64,
    "lat_max": 42.02,
    "lon_min": -87.95,
    "lon_max": -87.50,
}


def create_base_map(
    center: tuple[float, float] = CHICAGO_CENTER,
    zoom: int = 11,
) -> folium.Map:
    """Create base Chicago map.

    Args:
        center: Map center (lat, lon)
        zoom: Initial zoom level

    Returns:
        Folium Map object
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="cartodbpositron",
        control_scale=True,
    )

    # Add bounds rectangle for Chicago
    folium.Rectangle(
        bounds=[
            [CHICAGO_BOUNDS["lat_min"], CHICAGO_BOUNDS["lon_min"]],
            [CHICAGO_BOUNDS["lat_max"], CHICAGO_BOUNDS["lon_max"]],
        ],
        color="blue",
        weight=1,
        fill=False,
        opacity=0.3,
        popup="Chicago Crime Prediction Coverage Area",
    ).add_to(m)

    return m


def add_prediction_marker(
    m: folium.Map,
    result: PredictionResult,
) -> folium.Map:
    """Add a prediction marker to the map.

    Args:
        m: Folium map
        result: Prediction result

    Returns:
        Map with marker added
    """
    # Color based on predicted crime count
    if result.predicted_crimes < 10:
        color = "green"
        icon = "check"
    elif result.predicted_crimes < 30:
        color = "orange"
        icon = "info-sign"
    else:
        color = "red"
        icon = "warning-sign"

    popup_html = f"""
    <div style="font-family: Arial; min-width: 200px;">
        <h4 style="margin: 0 0 10px 0;">Crime Prediction</h4>
        <table style="width: 100%;">
            <tr><td><b>Predicted Crimes:</b></td><td>{result.predicted_crimes:.1f}</td></tr>
            <tr><td><b>Date:</b></td><td>{result.prediction_date}</td></tr>
            <tr><td><b>Location:</b></td><td>{result.latitude:.4f}, {result.longitude:.4f}</td></tr>
            {f'<tr><td><b>Grid ID:</b></td><td>{result.grid_id}</td></tr>' if result.grid_id else ''}
            {f'<tr><td><b>Confidence:</b></td><td>{result.confidence_lower:.1f} - {result.confidence_upper:.1f}</td></tr>' if result.confidence_lower else ''}
        </table>
    </div>
    """

    folium.Marker(
        location=[result.latitude, result.longitude],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color=color, icon=icon),
    ).add_to(m)

    return m


def add_hotspots_layer(
    m: folium.Map,
    hotspots: list[HotspotResult],
    show_markers: bool = True,
    show_heatmap: bool = True,
) -> folium.Map:
    """Add hotspots to the map.

    Args:
        m: Folium map
        hotspots: List of hotspot results
        show_markers: Whether to show individual markers
        show_heatmap: Whether to show heatmap layer

    Returns:
        Map with hotspots added
    """
    if not hotspots:
        return m

    # Add heatmap
    if show_heatmap:
        heat_data = [
            [h.latitude, h.longitude, h.predicted_crimes]
            for h in hotspots
        ]
        HeatMap(
            heat_data,
            min_opacity=0.4,
            radius=25,
            blur=15,
            gradient={
                0.2: "blue",
                0.4: "lime",
                0.6: "yellow",
                0.8: "orange",
                1.0: "red",
            },
        ).add_to(m)

    # Add markers for top hotspots
    if show_markers:
        for h in hotspots[:10]:  # Top 10 with markers
            color_map = {
                "critical": "darkred",
                "high": "red",
                "medium": "orange",
                "low": "green",
            }
            color = color_map.get(h.risk_level, "gray")

            popup_html = f"""
            <div style="font-family: Arial; min-width: 180px;">
                <h4 style="margin: 0 0 10px 0;">Hotspot #{h.rank}</h4>
                <table style="width: 100%;">
                    <tr><td><b>Predicted:</b></td><td>{h.predicted_crimes:.1f} crimes</td></tr>
                    <tr><td><b>Risk Level:</b></td><td style="color: {color}; font-weight: bold;">{h.risk_level.upper()}</td></tr>
                    <tr><td><b>Grid ID:</b></td><td>{h.grid_id}</td></tr>
                </table>
            </div>
            """

            folium.CircleMarker(
                location=[h.latitude, h.longitude],
                radius=8 + (11 - h.rank),  # Larger for higher ranked
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

            # Add rank label
            folium.Marker(
                location=[h.latitude, h.longitude],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 10px; color: white; font-weight: bold; background: {color}; border-radius: 50%; width: 20px; height: 20px; text-align: center; line-height: 20px;">{h.rank}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(m)

    return m


def add_grid_heatmap(
    m: folium.Map,
    grid_predictions: list[dict],
) -> folium.Map:
    """Add full grid heatmap to map.

    Args:
        m: Folium map
        grid_predictions: List of grid cell predictions

    Returns:
        Map with heatmap added
    """
    if not grid_predictions:
        return m

    heat_data = [
        [p["latitude"], p["longitude"], p["predicted_crimes"]]
        for p in grid_predictions
        if p.get("predicted_crimes", 0) > 0
    ]

    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            radius=20,
            blur=10,
            gradient={
                0.0: "green",
                0.25: "lime",
                0.5: "yellow",
                0.75: "orange",
                1.0: "red",
            },
        ).add_to(m)

    return m


# Export for React migration reference
MAP_CONFIG = {
    "center": CHICAGO_CENTER,
    "bounds": CHICAGO_BOUNDS,
    "default_zoom": 11,
    "tile_layer": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    "tile_attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    "heatmap_gradient": {
        "0.0": "#00ff00",
        "0.25": "#80ff00",
        "0.5": "#ffff00",
        "0.75": "#ff8000",
        "1.0": "#ff0000",
    },
}
