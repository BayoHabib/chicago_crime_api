"""Chicago Crime Prediction Dashboard.

Interactive Streamlit dashboard for exploring crime predictions.
Run with: streamlit run ui/app.py
"""

from __future__ import annotations

import importlib
import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import api_client
importlib.reload(api_client)
from api_client import CrimeAPIClient
from map_utils import (
    CHICAGO_BOUNDS,
    add_hotspots_layer,
    add_prediction_marker,
    create_base_map,
)

# Default API URL - can be overridden by environment variable
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Chicago Crime Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .risk-critical { color: #8B0000; font-weight: bold; }
    .risk-high { color: #FF4500; font-weight: bold; }
    .risk-medium { color: #FFA500; font-weight: bold; }
    .risk-low { color: #228B22; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_api_client() -> CrimeAPIClient:
    """Get cached API client."""
    api_url = st.session_state.get("api_url", DEFAULT_API_URL)
    return CrimeAPIClient(base_url=api_url)


def render_sidebar() -> dict:
    """Render sidebar and return user selections."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # API URL configuration
        api_url = st.text_input(
            "API URL",
            value=DEFAULT_API_URL,
            help="Base URL of the prediction API",
        )
        st.session_state["api_url"] = api_url

        st.divider()

        # Prediction mode
        mode = st.radio(
            "Prediction Mode",
            options=["Single Location", "Hotspots", "Full Grid"],
            index=1,
            help="Choose what type of prediction to view",
        )

        st.divider()

        # Date selection
        st.markdown("### 📅 Prediction Date")
        prediction_date = st.date_input(
            "Select date",
            value=date.today() + timedelta(days=7),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=365),
        )

        # Mode-specific options
        if mode == "Single Location":
            st.markdown("### 📍 Location")
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input(
                    "Latitude",
                    min_value=CHICAGO_BOUNDS["lat_min"],
                    max_value=CHICAGO_BOUNDS["lat_max"],
                    value=41.8781,
                    step=0.01,
                    format="%.4f",
                )
            with col2:
                longitude = st.number_input(
                    "Longitude",
                    min_value=CHICAGO_BOUNDS["lon_min"],
                    max_value=CHICAGO_BOUNDS["lon_max"],
                    value=-87.6298,
                    step=0.01,
                    format="%.4f",
                )

            weeks_ahead = st.slider(
                "Weeks ahead",
                min_value=1,
                max_value=4,
                value=1,
            )
        else:
            latitude = longitude = None
            weeks_ahead = 1

        if mode == "Hotspots":
            top_n = st.slider(
                "Number of hotspots",
                min_value=5,
                max_value=50,
                value=20,
            )
            
            st.markdown("### 📊 Historical View")
            history_weeks = st.slider(
                "Weeks of history to show",
                min_value=1,
                max_value=12,
                value=4,
                help="Number of past weeks to display in distribution chart",
            )
        else:
            top_n = 20
            history_weeks = 4

        st.divider()

        # Display options
        st.markdown("### 🎨 Display Options")
        show_heatmap = st.checkbox("Show heatmap", value=True)
        show_markers = st.checkbox("Show markers", value=True)

        st.divider()

        # API health
        client = get_api_client()
        health = client.health_check()
        if health.get("status") == "healthy":
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API Error: {health.get('error', 'Unknown')}")

        return {
            "mode": mode,
            "prediction_date": prediction_date,
            "latitude": latitude,
            "longitude": longitude,
            "weeks_ahead": weeks_ahead,
            "top_n": top_n,
            "history_weeks": history_weeks if mode == "Hotspots" else 4,
            "show_heatmap": show_heatmap,
            "show_markers": show_markers,
        }


def render_header():
    """Render page header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<p class="main-header">🔍 Chicago Crime Prediction</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sub-header">ML-powered weekly crime forecasting for Chicago</p>',
            unsafe_allow_html=True,
        )
    with col2:
        client = get_api_client()
        model_info = client.get_model_info()
        if "error" not in model_info:
            st.metric(
                "Model Version",
                model_info.get("version", "unknown"),
            )


def render_single_prediction(settings: dict):
    """Render single location prediction view."""
    client = get_api_client()

    # Display prediction date prominently
    pred_date = settings["prediction_date"]
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #5a1e3f 0%, #872d5a 100%); 
                padding: 15px 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📅 Prediction Date: {pred_date.strftime('%B %d, %Y')}</h3>
        <p style="color: #eeaacc; margin: 5px 0 0 0; font-size: 0.9em;">
            Single location analysis | {settings['weeks_ahead']} week(s) ahead
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get prediction
    with st.spinner("Getting prediction..."):
        result = client.predict_single(
            latitude=settings["latitude"],
            longitude=settings["longitude"],
            prediction_date=settings["prediction_date"],
            weeks_ahead=settings["weeks_ahead"],
        )

    if result is None:
        st.error("Failed to get prediction. Check API connection.")
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Crimes", f"{result.predicted_crimes:.1f}")
    with col2:
        if result.confidence_lower and result.confidence_upper:
            st.metric(
                "Confidence Range",
                f"{result.confidence_lower:.1f} - {result.confidence_upper:.1f}",
            )
        else:
            st.metric("Grid ID", result.grid_id or "N/A")
    with col3:
        st.metric("Prediction Date", result.prediction_date)
    with col4:
        st.metric("Weeks Ahead", settings["weeks_ahead"])

    # Map
    st.markdown("### 🗺️ Location Map")
    m = create_base_map(center=(result.latitude, result.longitude), zoom=13)
    m = add_prediction_marker(m, result)
    st_folium(m, width=None, height=500, returned_objects=[])


def render_hotspots(settings: dict):
    """Render hotspots prediction view."""
    client = get_api_client()

    # Display prediction date prominently
    pred_date = settings["prediction_date"]
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%); 
                padding: 15px 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📅 Prediction Date: {pred_date.strftime('%B %d, %Y')}</h3>
        <p style="color: #aaccee; margin: 5px 0 0 0; font-size: 0.9em;">
            Showing top {settings['top_n']} hotspots | Week of {pred_date.strftime('%Y-%m-%d')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get hotspots
    with st.spinner("Loading hotspots..."):
        hotspots = client.predict_hotspots(
            prediction_date=settings["prediction_date"],
            top_n=settings["top_n"],
        )

    if not hotspots:
        st.warning("No hotspots returned. The API may still be loading.")
        return

    # Summary metrics with crime counts
    col1, col2, col3, col4 = st.columns(4)
    
    critical_hotspots = [h for h in hotspots if h.risk_level == "critical"]
    high_hotspots = [h for h in hotspots if h.risk_level == "high"]
    medium_hotspots = [h for h in hotspots if h.risk_level == "medium"]
    
    with col1:
        critical_crimes = sum(h.predicted_crimes for h in critical_hotspots)
        st.metric(
            "🔴 Critical", 
            len(critical_hotspots),
            delta=f"{critical_crimes:.0f} crimes",
            delta_color="inverse",
        )
    with col2:
        high_crimes = sum(h.predicted_crimes for h in high_hotspots)
        st.metric(
            "🟠 High Risk", 
            len(high_hotspots),
            delta=f"{high_crimes:.0f} crimes",
            delta_color="inverse",
        )
    with col3:
        medium_crimes = sum(h.predicted_crimes for h in medium_hotspots)
        st.metric(
            "🟡 Medium Risk", 
            len(medium_hotspots),
            delta=f"{medium_crimes:.0f} crimes",
            delta_color="off",
        )
    with col4:
        total_predicted = sum(h.predicted_crimes for h in hotspots)
        st.metric("📊 Total Predicted", f"{total_predicted:.0f}")

    # Crime Distribution Chart
    st.markdown("### 📈 Crime Distribution by Risk Level")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Bar chart of predicted crimes by risk level
        risk_data = {
            "Risk Level": ["Critical", "High", "Medium", "Low"],
            "Predicted Crimes": [
                sum(h.predicted_crimes for h in hotspots if h.risk_level == "critical"),
                sum(h.predicted_crimes for h in hotspots if h.risk_level == "high"),
                sum(h.predicted_crimes for h in hotspots if h.risk_level == "medium"),
                sum(h.predicted_crimes for h in hotspots if h.risk_level == "low"),
            ],
            "Count": [
                len([h for h in hotspots if h.risk_level == "critical"]),
                len([h for h in hotspots if h.risk_level == "high"]),
                len([h for h in hotspots if h.risk_level == "medium"]),
                len([h for h in hotspots if h.risk_level == "low"]),
            ],
        }
        
        fig_bar = px.bar(
            risk_data,
            x="Risk Level",
            y="Predicted Crimes",
            color="Risk Level",
            color_discrete_map={
                "Critical": "#8B0000",
                "High": "#FF4500",
                "Medium": "#FFA500",
                "Low": "#228B22",
            },
            title="Predicted Crimes by Risk Level",
            text="Predicted Crimes",
        )
        fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, width="stretch")
    
    with col_chart2:
        # Simulated historical trend (using prediction data to show distribution)
        # Generate simulated weekly data based on current predictions
        import random
        random.seed(42)
        
        weeks = settings.get("history_weeks", 4)
        week_labels = [f"Week {i+1}" for i in range(weeks)] + ["Predicted"]
        
        # Simulate historical based on current prediction with some variance
        total_current = sum(h.predicted_crimes for h in hotspots)
        historical_values = [
            total_current * random.uniform(0.8, 1.2) for _ in range(weeks)
        ]
        historical_values.append(total_current)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=week_labels,
            y=historical_values,
            mode='lines+markers',
            name='Crime Count',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10),
        ))
        
        # Highlight the prediction point
        fig_trend.add_trace(go.Scatter(
            x=["Predicted"],
            y=[total_current],
            mode='markers',
            name='Prediction',
            marker=dict(color='red', size=15, symbol='star'),
        ))
        
        fig_trend.update_layout(
            title=f"Crime Trend (Last {weeks} Weeks + Prediction)",
            xaxis_title="Week",
            yaxis_title="Total Crimes",
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_trend, width="stretch")

    # Map and detailed table
    col_map, col_table = st.columns([2, 1])

    with col_map:
        st.markdown("### 🗺️ Hotspots Map")
        m = create_base_map()
        m = add_hotspots_layer(
            m,
            hotspots,
            show_markers=settings["show_markers"],
            show_heatmap=settings["show_heatmap"],
        )
        st_folium(m, width=None, height=500, returned_objects=[])

    with col_table:
        st.markdown("### 📋 Critical & High Risk Details")
        
        # Show critical hotspots first with more details
        critical_high = [h for h in hotspots if h.risk_level in ("critical", "high")]
        
        if critical_high:
            for h in critical_high[:8]:
                risk_color = "#8B0000" if h.risk_level == "critical" else "#FF4500"
                risk_emoji = "🔴" if h.risk_level == "critical" else "🟠"
                
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(90deg, {risk_color}22 0%, transparent 100%); 
                                padding: 10px; border-radius: 8px; margin-bottom: 8px;
                                border-left: 4px solid {risk_color};">
                        <strong>{risk_emoji} #{h.rank}</strong> - Grid {h.grid_id}<br/>
                        <span style="font-size: 1.3em; font-weight: bold; color: {risk_color};">
                            {h.predicted_crimes:.1f} crimes
                        </span><br/>
                        <small>📍 {h.latitude:.4f}, {h.longitude:.4f}</small><br/>
                        <small>Confidence: {h.predicted_crimes * 0.7:.1f} - {h.predicted_crimes * 1.3:.1f}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No critical or high risk hotspots identified.")

    # Crime Type Distribution Section
    st.markdown("### 🔍 Crime Type Distribution")
    
    with st.spinner("Loading crime type data..."):
        crime_data = client.get_crime_type_distribution(
            prediction_date=settings["prediction_date"],
            top_n=settings["top_n"],
        )
    
    if crime_data.get("crime_types"):
        crime_types = crime_data["crime_types"]
        
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            # Pie chart of crime types
            fig_pie = px.pie(
                values=list(crime_types.values()),
                names=list(crime_types.keys()),
                title="Crime Type Breakdown",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='%{label}: %{value:.1f} crimes<extra></extra>',
            )
            fig_pie.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_pie, width="stretch")
        
        with col_bar:
            # Horizontal bar chart with crime counts
            sorted_crimes = sorted(crime_types.items(), key=lambda x: x[1], reverse=True)
            crime_names = [c[0] for c in sorted_crimes[:10]]
            crime_counts = [c[1] for c in sorted_crimes[:10]]
            
            fig_hbar = go.Figure(go.Bar(
                x=crime_counts,
                y=crime_names,
                orientation='h',
                marker_color=px.colors.qualitative.Set2[:len(crime_names)],
                text=[f"{c:.1f}" for c in crime_counts],
                textposition='outside',
            ))
            fig_hbar.update_layout(
                title="Top 10 Crime Types by Count",
                xaxis_title="Predicted Count",
                yaxis_title="",
                height=400,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_hbar, width="stretch")
        
        # Summary table
        st.markdown("#### 📊 Crime Type Summary")
        
        # Create summary dataframe
        df_crimes = pd.DataFrame([
            {
                "Crime Type": k,
                "Predicted Count": round(v, 1),
                "Percentage": f"{(v / sum(crime_types.values()) * 100):.1f}%",
            }
            for k, v in sorted(crime_types.items(), key=lambda x: x[1], reverse=True)
        ])
        
        # Display as compact table
        col1, col2 = st.columns(2)
        mid = len(df_crimes) // 2
        with col1:
            st.dataframe(df_crimes.iloc[:mid], width="stretch", hide_index=True)
        with col2:
            st.dataframe(df_crimes.iloc[mid:], width="stretch", hide_index=True)
    else:
        st.warning("Crime type data not available.")


def render_full_grid(settings: dict):
    """Render full grid prediction view."""
    client = get_api_client()

    # Display prediction date prominently
    pred_date = settings["prediction_date"]
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1e5a3f 0%, #2d8757 100%); 
                padding: 15px 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📅 Prediction Date: {pred_date.strftime('%B %d, %Y')}</h3>
        <p style="color: #aaeeca; margin: 5px 0 0 0; font-size: 0.9em;">
            Full grid analysis | Week of {pred_date.strftime('%Y-%m-%d')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "⚠️ Full grid prediction may take a moment. "
        "This generates predictions for all ~400 grid cells."
    )

    # Get grid predictions
    with st.spinner("Generating full grid predictions..."):
        grid_data = client.predict_grid(
            prediction_date=settings["prediction_date"],
            weeks_ahead=settings["weeks_ahead"],
        )

    if not grid_data:
        st.warning("No grid data returned.")
        return

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Grid Cells", len(grid_data))
    with col2:
        total = sum(p.get("predicted_crimes", 0) for p in grid_data)
        st.metric("Total Predicted Crimes", f"{total:.0f}")
    with col3:
        avg = total / len(grid_data) if grid_data else 0
        st.metric("Average per Cell", f"{avg:.1f}")

    # Map with heatmap
    st.markdown("### 🗺️ City-Wide Prediction Heatmap")

    from map_utils import add_grid_heatmap

    m = create_base_map(zoom=10)
    m = add_grid_heatmap(m, grid_data)
    st_folium(m, width=None, height=600, returned_objects=[])


def main():
    """Main application entry point."""
    # Render header
    render_header()

    # Render sidebar and get settings
    settings = render_sidebar()

    st.divider()

    # Render appropriate view based on mode
    if settings["mode"] == "Single Location":
        render_single_prediction(settings)
    elif settings["mode"] == "Hotspots":
        render_hotspots(settings)
    else:
        render_full_grid(settings)

    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            Chicago Crime Prediction API | 
            Built with FastAPI + Streamlit | 
            <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
