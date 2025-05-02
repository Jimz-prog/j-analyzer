import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# Page Configuration
st.set_page_config(page_title="<myJ>Analyzer Combustion Analysis Tool", layout="wide")
st.title("🔥 <myJ>Combustion Analysis Tool")
st.sidebar.image("janalyzer.png", width=600)

# Constants
DEFAULT_BORE = 60.0  # mm
DEFAULT_STROKE = 43.0  # mm
DEFAULT_CON_ROD = 87.0  # mm
DEFAULT_COMPRESSION_RATIO = 8.5
DEFAULT_IGNITION_TIMING = -10.0  # degrees BTDC
GAMMA = 1.35
DEFAULT_DIESEL_START_OF_INJECTION = -15.0  # degrees BTDC
DEFAULT_DIESEL_INJECTION_DURATION = 20.0  # degrees
DIESEL_GAMMA = 1.30  # Specific heat ratio for diesel
DEFAULT_KNOCK_THRESHOLD = 0.5  # bar/degree
KNOCK_WINDOW_SIZE = 10  # degrees
KNOCK_SENSITIVITY = 1.5  # Factor of standard deviation

# Helper Functions
def calculate_volume(theta_deg, stroke, con_rod, compression_ratio, V_clearance):
    """Calculate cylinder volume at given crank angle"""
    theta = np.radians(theta_deg)
    r = stroke / 2
    a = con_rod
    R = a / r
    term = 1 + 0.5 * (compression_ratio - 1) * (
        (1 - np.cos(theta)) + (1 / R) * (1 - np.sqrt(1 - (np.sin(theta) ** 2 / R**2))))
    return V_clearance * term

def calculate_combustion_metrics(
    theta,
    pressure,
    time_sec,
    gamma,
    V_disp,
    stroke,
    con_rod,
    compression_ratio,
    V_clearance,
    is_diesel=False,
    soi=None,
    injection_duration=None,
    knock_threshold=None
):
    """Calculate combustion metrics for a cycle"""
    pressure_smooth = savgol_filter(pressure, window_length=5, polyorder=2)
    dp_dtheta = np.gradient(pressure_smooth, theta)
    
    # Initialize d2p_dtheta2 with zeros
    d2p_dtheta2 = np.zeros_like(dp_dtheta)
    if len(theta) > 1:  # Only calculate if we have enough data points
        d2p_dtheta2 = np.gradient(dp_dtheta, theta)
    
    valid_mask = (pressure_smooth > 0) & (~np.isnan(pressure_smooth)) & (~np.isnan(dp_dtheta))
    HRR = np.zeros_like(pressure_smooth)
    HRR[valid_mask] = (gamma / (gamma - 1)) * pressure_smooth[valid_mask] * dp_dtheta[valid_mask]
    
    cumulative_HRR = np.cumsum(HRR)
    if np.max(np.abs(cumulative_HRR)) > 1e-6:
        CA50_index = np.argmin(np.abs((cumulative_HRR / cumulative_HRR[-1]) - 0.5))
        CA50 = theta[CA50_index]
    else:
        CA50 = np.nan
    
    V = calculate_volume(theta, stroke, con_rod, compression_ratio, V_clearance)
    IMEP = np.trapz(pressure_smooth, V) / V_disp
    
    # Knocking analysis
    knock_results = {
        'knock_intensity': 0,
        'knock_angles': [],
        'knock_pressures': [],
        'is_knocking': False,
        'd2p_dtheta2': d2p_dtheta2  # Ensure this is always available
    }
    
    if len(theta) > KNOCK_WINDOW_SIZE:  # Only calculate if we have enough data
        rolling_std = pd.Series(d2p_dtheta2).rolling(window=KNOCK_WINDOW_SIZE, center=True).std().values
        if knock_threshold is None:
            knock_threshold = KNOCK_SENSITIVITY * np.nanmean(rolling_std)
        
        knock_indices = np.where(np.abs(d2p_dtheta2) > knock_threshold)[0]
        knock_mask = (theta >= -20) & (theta <= 60)  # Only consider knocking in typical range
        knock_indices = knock_indices[knock_mask[knock_indices]] if len(knock_indices) > 0 else []
        
        if len(knock_indices) > 0:
            knock_results.update({
                'knock_intensity': np.max(np.abs(d2p_dtheta2[knock_indices])),
                'knock_angles': theta[knock_indices],
                'knock_pressures': pressure_smooth[knock_indices],
                'is_knocking': True
            })
    
    # Diesel-specific calculations
    diesel_metrics = {}
    if is_diesel and soi is not None and injection_duration is not None:
        injection_start = soi
        injection_end = soi + injection_duration
        
        combustion_start_threshold = 0.05 * np.max(HRR)
        combustion_start_idx = np.where(HRR > combustion_start_threshold)[0]
        if len(combustion_start_idx) > 0:
            combustion_start = theta[combustion_start_idx[0]]
            ignition_delay = combustion_start - soi
        else:
            combustion_start = np.nan
            ignition_delay = np.nan
        
        if not np.isnan(combustion_start):
            premixed_end_idx = np.argmax(HRR)
            premixed_end = theta[premixed_end_idx]
            diffusion_end = theta[np.argmin(np.abs(cumulative_HRR - 0.95*cumulative_HRR[-1]))]
            
            total_combustion_duration = diffusion_end - combustion_start
            premixed_combustion_ratio = (premixed_end - combustion_start) / total_combustion_duration
            
            diesel_metrics = {
                'injection_start': injection_start,
                'injection_end': injection_end,
                'combustion_start': combustion_start,
                'ignition_delay': ignition_delay,
                'premixed_start': combustion_start,
                'premixed_end': premixed_end,
                'diffusion_start': premixed_end,
                'diffusion_end': diffusion_end,
                'max_hrr_angle': premixed_end,
                'max_hrr_value': np.max(HRR),
                'premixed_ratio': premixed_combustion_ratio,
                'total_combustion_duration': total_combustion_duration
            }
    
    return {
        'pressure': pressure_smooth,
        'dp_dtheta': dp_dtheta,
        'HRR': HRR,
        'cumulative_HRR': cumulative_HRR,
        'CA50': CA50,
        'IMEP': IMEP,
        'volume': V,
        'time_sec': time_sec,
        'knock': knock_results,
        'diesel_metrics': diesel_metrics if is_diesel else None
    }

# File Uploader
uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [col.lower().strip() for col in df.columns]

        required_cols = {'cycle', 'crank_angle', 'pressure', 'timestamps'}
        if not required_cols.issubset(df.columns):
            st.error("❌ File must contain columns: timestamps, cycle, crank_angle, pressure")
        else:
            # Data Preprocessing
            df['time_sec'] = (df['timestamps'] - df['timestamps'].iloc[0]) * 1e-7
            df = df.sort_values(by=['cycle', 'crank_angle'])
            
            # RPM Calculation
            df['dt'] = df.groupby('cycle')['time_sec'].diff().fillna(0)
            df['dtheta'] = df.groupby('cycle')['crank_angle'].diff().fillna(0)
            df['rpm_calc'] = np.where(df['dt'] != 0, (df['dtheta'] / df['dt']) * (60 / 360), 0)
            df['rpm_calc'] = df['rpm_calc'].replace([np.inf, -np.inf], 0).fillna(0)
            rpm_per_cycle = df.groupby('cycle')['rpm_calc'].mean().reset_index().rename(columns={'rpm_calc': 'RPM'})

            st.subheader("📌 Average RPM per Cycle")
            st.dataframe(rpm_per_cycle.round(1))

            # Engine Parameters
            st.subheader("🔧 Engine Parameters")
            with st.expander("Set Engine Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    bore = st.number_input("Bore (mm)", min_value=1.0, value=DEFAULT_BORE) / 1000
                    stroke = st.number_input("Stroke (mm)", min_value=1.0, value=DEFAULT_STROKE) / 1000
                    con_rod = st.number_input("Connecting Rod Length (mm)", min_value=1.0, value=DEFAULT_CON_ROD) / 1000
                with col2:
                    compression_ratio = st.number_input("Compression Ratio", min_value=1.0, value=DEFAULT_COMPRESSION_RATIO)
                    ignition_timing = st.number_input("Ignition Timing (degrees BTDC)", 
                                                   min_value=-90.0, 
                                                   max_value=90.0, 
                                                   value=DEFAULT_IGNITION_TIMING)
                
                engine_type = st.radio("Engine Type", ["Gasoline", "Diesel"], index=0)
                is_diesel = (engine_type == "Diesel")
                
                if is_diesel:
                    st.markdown("**Diesel Engine Parameters**")
                    col1, col2 = st.columns(2)
                    with col1:
                        start_of_injection = st.number_input("Start of Injection (degrees BTDC)", 
                                                          min_value=-90.0, 
                                                          max_value=90.0, 
                                                          value=DEFAULT_DIESEL_START_OF_INJECTION)
                    with col2:
                        injection_duration = st.number_input("Injection Duration (degrees)", 
                                                          min_value=0.1, 
                                                          max_value=180.0, 
                                                          value=DEFAULT_DIESEL_INJECTION_DURATION)
                    gamma = DIESEL_GAMMA
                else:
                    gamma = GAMMA
                
                use_smoothed = st.checkbox("Use Smoothed Data", value=True)
                knock_threshold = st.number_input("Knock Threshold (bar/deg²)", 
                                                min_value=0.1, 
                                                max_value=5.0, 
                                                value=DEFAULT_KNOCK_THRESHOLD, 
                                                step=0.1,
                                                help="Threshold for detecting pressure oscillations characteristic of knock")

            # Engine Constants
            V_disp = (np.pi / 4) * bore**2 * stroke
            V_clearance = V_disp / (compression_ratio - 1)

            # Tabs for Analysis
            tab1, tab2 = st.tabs(["📊 Single Cycle Analysis", "📈 Multi-Cycle Analysis"])
            
            with tab1:
                st.subheader("📊 Single Cycle Analysis")
                
                available_cycles = df['cycle'].dropna().unique()
                selected_cycle = st.selectbox("Select Cycle", sorted(available_cycles), key="single_cycle_select")
                df_cycle = df[df['cycle'] == selected_cycle]
                theta = df_cycle['crank_angle'].to_numpy()
                pressure = df_cycle['pressure'].to_numpy()
                time_sec = df_cycle['time_sec'].to_numpy()
                
                if len(theta) < 2 or np.all(np.isnan(pressure)):
                    st.warning("⚠️ Insufficient data for selected cycle.")
                else:
                    results = calculate_combustion_metrics(
                        theta=theta,
                        pressure=pressure,
                        time_sec=time_sec,
                        gamma=gamma,
                        V_disp=V_disp,
                        stroke=stroke,
                        con_rod=con_rod,
                        compression_ratio=compression_ratio,
                        V_clearance=V_clearance,
                        is_diesel=is_diesel,
                        soi=start_of_injection if is_diesel else None,
                        injection_duration=injection_duration if is_diesel else None,
                        knock_threshold=knock_threshold
                    )
                    
                    # Create tabs for different plots
                    tab_names = ["Pressure vs Angle", "Pressure vs Time", "dP/dθ", "HRR", "PV Diagram", "Knock Analysis"]
                    if is_diesel:
                        tab_names.insert(5, "Diesel Analysis")
                    tabs = st.tabs(tab_names)
                    
                    with tabs[0]:  # Pressure vs Angle
                        fig_pressure_angle = go.Figure()
                        fig_pressure_angle.add_trace(go.Scatter(
                            x=theta,
                            y=results['pressure'],
                            mode='lines',
                            name='Pressure',
                            line=dict(color='blue'),
                            hovertemplate='Angle: %{x}°<br>Pressure: %{y:.2f} bar<extra></extra>'
                        ))
                        fig_pressure_angle.add_vline(
                            x=ignition_timing,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Ignition: {ignition_timing}° BTDC",
                            annotation_position="top right"
                        )
                        fig_pressure_angle.update_layout(
                            title=f"Pressure vs Crank Angle - Cycle {selected_cycle}",
                            xaxis_title="Crank Angle (deg)",
                            yaxis_title="Pressure (bar)",
                            hovermode="x unified",
                            showlegend=True
                        )
                        st.plotly_chart(fig_pressure_angle, use_container_width=True)
                    
                    with tabs[1]:  # Pressure vs Time
                        fig_pressure_time = go.Figure()
                        fig_pressure_time.add_trace(go.Scatter(
                            x=results['time_sec'],
                            y=results['pressure'],
                            mode='lines',
                            name='Pressure',
                            line=dict(color='blue'),
                            hovertemplate='Time: %{x:.4f} s<br>Pressure: %{y:.2f} bar<extra></extra>'
                        ))
                        
                        ignition_idx = np.argmin(np.abs(theta - ignition_timing))
                        if ignition_idx < len(results['time_sec']):
                            fig_pressure_time.add_trace(go.Scatter(
                                x=[results['time_sec'][ignition_idx]],
                                y=[results['pressure'][ignition_idx]],
                                mode='markers',
                                name='Ignition Point',
                                marker=dict(color='red', size=10),
                                hovertemplate='Ignition Time: %{x:.4f} s<br>Pressure: %{y:.2f} bar<extra></extra>'
                            ))
                        
                        fig_pressure_time.update_layout(
                            title=f"Pressure vs Time - Cycle {selected_cycle}",
                            xaxis_title="Time (s)",
                            yaxis_title="Pressure (bar)",
                            hovermode="x unified",
                            showlegend=True
                        )
                        st.plotly_chart(fig_pressure_time, use_container_width=True)
                    
                    with tabs[2]:  # dP/dθ
                        fig_dp = go.Figure()
                        fig_dp.add_trace(go.Scatter(
                            x=theta,
                            y=results['dp_dtheta'],
                            mode='lines',
                            name='dP/dθ',
                            line=dict(color='green'),
                            hovertemplate='Angle: %{x}°<br>dP/dθ: %{y:.4f} bar/deg<extra></extra>'
                        ))
                        fig_dp.update_layout(
                            title=f"dP/dθ vs Crank Angle - Cycle {selected_cycle}",
                            xaxis_title="Crank Angle (deg)",
                            yaxis_title="dP/dθ (bar/deg)",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_dp, use_container_width=True)
                    
                    with tabs[3]:  # HRR
                        fig_hrr = go.Figure()
                        fig_hrr.add_trace(go.Scatter(
                            x=theta,
                            y=results['HRR'],
                            mode='lines',
                            name='HRR',
                            line=dict(color='purple'),
                            hovertemplate='Angle: %{x}°<br>HRR: %{y:.2f} J/deg<extra></extra>'
                        ))
                        fig_hrr.update_layout(
                            title=f"Heat Release Rate - Cycle {selected_cycle}",
                            xaxis_title="Crank Angle (deg)",
                            yaxis_title="HRR (J/deg)",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_hrr, use_container_width=True)
                    
                    with tabs[4]:  # PV Diagram
                        fig_pv = go.Figure()
                        fig_pv.add_trace(go.Scatter(
                            x=results['volume'] * 1e6,
                            y=results['pressure'],
                            mode='lines',
                            name='PV Curve',
                            line=dict(color='orange'),
                            hovertemplate='Volume: %{x:.2f} cc<br>Pressure: %{y:.2f} bar<extra></extra>'
                        ))
                        fig_pv.update_layout(
                            title=f"PV Diagram - Cycle {selected_cycle}",
                            xaxis_title="Volume (cc)",
                            yaxis_title="Pressure (bar)",
                            hovermode="closest"
                        )
                        st.plotly_chart(fig_pv, use_container_width=True)
                    
                    if is_diesel and len(tabs) > 5:  # Diesel Analysis
                        with tabs[5]:
                            st.subheader("⛽ Diesel Combustion Analysis")
                            
                            if results.get('diesel_metrics'):
                                dm = results['diesel_metrics']
                                
                                fig_diesel = go.Figure()
                                fig_diesel.add_trace(go.Scatter(
                                    x=theta,
                                    y=results['HRR'],
                                    mode='lines',
                                    name='HRR',
                                    line=dict(color='blue'),
                                    hovertemplate='Angle: %{x}°<br>HRR: %{y:.2f} J/deg<extra></extra>'
                                ))
                                
                                fig_diesel.add_vrect(
                                    x0=dm['injection_start'],
                                    x1=dm['injection_end'],
                                    fillcolor="lightblue",
                                    opacity=0.3,
                                    layer="below",
                                    line_width=0,
                                    annotation_text="Injection",
                                    annotation_position="top left"
                                )
                                
                                if not np.isnan(dm['combustion_start']):
                                    fig_diesel.add_vrect(
                                        x0=dm['premixed_start'],
                                        x1=dm['premixed_end'],
                                        fillcolor="green",
                                        opacity=0.2,
                                        layer="below",
                                        line_width=0,
                                        annotation_text="Premixed",
                                        annotation_position="top left"
                                    )
                                    
                                    fig_diesel.add_vrect(
                                        x0=dm['diffusion_start'],
                                        x1=dm['diffusion_end'],
                                        fillcolor="orange",
                                        opacity=0.2,
                                        layer="below",
                                        line_width=0,
                                        annotation_text="Diffusion",
                                        annotation_position="top left"
                                    )
                                    
                                    fig_diesel.add_vline(
                                        x=dm['combustion_start'],
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"Combustion Start: {dm['combustion_start']:.1f}°",
                                        annotation_position="top right"
                                    )
                                    
                                    fig_diesel.add_vline(
                                        x=dm['max_hrr_angle'],
                                        line_dash="dash",
                                        line_color="purple",
                                        annotation_text=f"Peak HRR: {dm['max_hrr_angle']:.1f}°",
                                        annotation_position="top right"
                                    )
                                
                                fig_diesel.update_layout(
                                    title=f"Diesel Combustion Phases - Cycle {selected_cycle}",
                                    xaxis_title="Crank Angle (deg)",
                                    yaxis_title="HRR (J/deg)",
                                    hovermode="x unified"
                                )
                                st.plotly_chart(fig_diesel, use_container_width=True)
                                
                                st.subheader("Diesel Combustion Metrics")
                                cols = st.columns(3)
                                with cols[0]:
                                    st.metric("Start of Injection", f"{start_of_injection:.1f}° BTDC")
                                    if not np.isnan(dm.get('ignition_delay', np.nan)):
                                        st.metric("Ignition Delay", f"{dm['ignition_delay']:.2f}°")
                                with cols[1]:
                                    if not np.isnan(dm.get('combustion_start', np.nan)):
                                        st.metric("Combustion Start", f"{dm['combustion_start']:.1f}°")
                                        st.metric("Premixed Duration", f"{dm['premixed_end']-dm['premixed_start']:.2f}°")
                                with cols[2]:
                                    if not np.isnan(dm.get('combustion_start', np.nan)):
                                        st.metric("Diffusion Duration", f"{dm['diffusion_end']-dm['diffusion_start']:.2f}°")
                                        st.metric("Premixed Ratio", f"{dm['premixed_ratio']*100:.1f}%")
                            else:
                                st.warning("Could not calculate diesel combustion metrics for this cycle.")
                    
                    with tabs[-1 if not is_diesel else 6]:  # Knock Analysis
                        st.subheader("🛎️ Knock Analysis")
                        
                        fig_knock = go.Figure()
                        fig_knock.add_trace(go.Scatter(
                            x=theta,
                            y=results['pressure'],
                            mode='lines',
                            name='Pressure',
                            line=dict(color='blue'),
                            hovertemplate='Angle: %{x}°<br>Pressure: %{y:.2f} bar<extra></extra>'
                        ))
                        
                        if results['knock']['is_knocking']:
                            fig_knock.add_trace(go.Scatter(
                                x=results['knock']['knock_angles'],
                                y=results['knock']['knock_pressures'],
                                mode='markers',
                                name='Knock Events',
                                marker=dict(color='red', size=8),
                                hovertemplate='Knock at: %{x}°<br>Pressure: %{y:.2f} bar<extra></extra>'
                            ))
                            
                            fig_knock.add_trace(go.Scatter(
                                x=theta,
                                y=results['knock']['d2p_dtheta2'],
                                mode='lines',
                                name='d²P/dθ²',
                                line=dict(color='green'),
                                yaxis='y2',
                                hovertemplate='Angle: %{x}°<br>d²P/dθ²: %{y:.4f} bar/deg²<extra></extra>'
                            ))
                            
                                                      
                            fig_knock.update_layout(
                                yaxis2=dict(
                                    title="d²P/dθ² (bar/deg²)",
                                    overlaying='y',
                                    side='right'
                                )
                            )
                            
                            st.success(f"⚠️ Knock detected! Maximum intensity: {results['knock']['knock_intensity']:.2f} bar/deg²")
                        else:
                            st.info("✅ No knocking detected in this cycle")
                        
                        fig_knock.add_vrect(
                            x0=-20, x1=60,
                            fillcolor="red",
                            opacity=0.1,
                            layer="above",
                            line_width=0,
                            annotation_text="Knock Window",
                            annotation_position="top right",
                            annotation_font_size=12,
                            annotation_font_color="blue"
                        )
                        fig_knock.add_hrect(
                            y0=-knock_threshold,
                            y1=knock_threshold,
                            fillcolor="orange",
                            opacity=0.15,
                            layer="above",
                            line_width=0,
                            annotation_text=f"±{knock_threshold:.2f} bar/deg²",
                            annotation_position="top left",
                            annotation_font_size=12,
                            annotation_font_color="orange",
                            yref="y2"
                        )

                        fig_knock.update_layout(
                            title=f"Knock Analysis - Cycle {selected_cycle}",
                            xaxis_title="Crank Angle (deg)",
                            yaxis_title="Pressure (bar)",
                            hovermode="x unified",
                            showlegend=True,
                            xaxis=dict(range=[-20, 60])  # Zoom to knock window
                        )

                        st.plotly_chart(fig_knock, use_container_width=True)
                        
                        st.subheader("Knock Metrics")
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Knock Detected", "Yes" if results['knock']['is_knocking'] else "No")
                            st.metric("Number of Knock Events", len(results['knock']['knock_angles']))
                        with cols[1]:
                            st.metric("Max Knock Intensity", f"{results['knock']['knock_intensity']:.2f} bar/deg²")
                            st.metric("Knock Threshold", f"{knock_threshold:.2f} bar/deg²")
                    
                    # Analysis Results
                    st.subheader("📍 Analysis Results")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Cycle Number", selected_cycle)
                        st.metric("CA50", f"{results['CA50']:.2f}°")
                    with cols[1]:
                        st.metric("IMEP", f"{results['IMEP']:.2f} bar")
                        st.metric("Peak Pressure", f"{np.max(results['pressure']):.2f} bar")
                    with cols[2]:
                        if ignition_idx < len(results['time_sec']):
                            st.metric("Ignition Time", f"{results['time_sec'][ignition_idx]:.4f} s")
                        st.metric("Compression Ratio", f"{compression_ratio:.1f}")
            
            with tab2:
                st.subheader("📈 Multi-Cycle Analysis")
                
                mtab_names = ["Pressure vs Angle", "Pressure vs Time", "dP/dθ", "HRR", "PV Diagrams"]
                mtabs = st.tabs(mtab_names)
                
                with mtabs[0]:  # Pressure vs Angle (All Cycles)
                    fig_pressure_angle_all = go.Figure()
                    for c in sorted(df['cycle'].dropna().unique()):
                        df_c = df[df['cycle'] == c]
                        theta = df_c['crank_angle'].to_numpy()
                        pressure = df_c['pressure'].to_numpy()
                        time_sec = df_c['time_sec'].to_numpy()
                        
                        if len(theta) >= 2:
                            results = calculate_combustion_metrics(
                                theta=theta,
                                pressure=pressure,
                                time_sec=time_sec,
                                gamma=gamma,
                                V_disp=V_disp,
                                stroke=stroke,
                                con_rod=con_rod,
                                compression_ratio=compression_ratio,
                                V_clearance=V_clearance,
                                is_diesel=is_diesel,
                                soi=start_of_injection if is_diesel else None,
                                injection_duration=injection_duration if is_diesel else None,
                                knock_threshold=knock_threshold
                            )
                            fig_pressure_angle_all.add_trace(go.Scatter(
                                x=theta,
                                y=results['pressure'],
                                mode='lines',
                                name=f'Cycle {c}',
                                hovertemplate='Angle: %{x}°<br>Pressure: %{y:.2f} bar<extra></extra>'
                            ))
                    
                    fig_pressure_angle_all.add_vline(
                        x=ignition_timing,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Ignition: {ignition_timing}° BTDC",
                        annotation_position="top right"
                    )
                    fig_pressure_angle_all.update_layout(
                        title="Pressure vs Crank Angle (All Cycles)",
                        xaxis_title="Crank Angle (deg)",
                        yaxis_title="Pressure (bar)",
                        hovermode="x unified",
                        showlegend=True
                    )
                    st.plotly_chart(fig_pressure_angle_all, use_container_width=True)
                
                with mtabs[1]:  # Pressure vs Time (All Cycles)
                    fig_pressure_time_all = go.Figure()
                    for c in sorted(df['cycle'].dropna().unique()):
                        df_c = df[df['cycle'] == c]
                        theta = df_c['crank_angle'].to_numpy()
                        pressure = df_c['pressure'].to_numpy()
                        time_sec = df_c['time_sec'].to_numpy()
                        
                        if len(theta) >= 2:
                            results = calculate_combustion_metrics(
                                theta=theta,
                                pressure=pressure,
                                time_sec=time_sec,
                                gamma=gamma,
                                V_disp=V_disp,
                                stroke=stroke,
                                con_rod=con_rod,
                                compression_ratio=compression_ratio,
                                V_clearance=V_clearance,
                                is_diesel=is_diesel,
                                soi=start_of_injection if is_diesel else None,
                                injection_duration=injection_duration if is_diesel else None,
                                knock_threshold=knock_threshold
                            )
                            fig_pressure_time_all.add_trace(go.Scatter(
                                x=results['time_sec'],
                                y=results['pressure'],
                                mode='lines',
                                name=f'Cycle {c}',
                                hovertemplate='Time: %{x:.4f} s<br>Pressure: %{y:.2f} bar<extra></extra>'
                            ))
                    
                    fig_pressure_time_all.update_layout(
                        title="Pressure vs Time (All Cycles)",
                        xaxis_title="Time (s)",
                        yaxis_title="Pressure (bar)",
                        hovermode="x unified",
                        showlegend=True
                    )
                    st.plotly_chart(fig_pressure_time_all, use_container_width=True)
                
                with mtabs[2]:  # dP/dθ (All Cycles)
                    fig_dp_all = go.Figure()
                    for c in sorted(df['cycle'].dropna().unique()):
                        df_c = df[df['cycle'] == c]
                        theta = df_c['crank_angle'].to_numpy()
                        pressure = df_c['pressure'].to_numpy()
                        time_sec = df_c['time_sec'].to_numpy()
                        
                        if len(theta) >= 2:
                            results = calculate_combustion_metrics(
                                theta=theta,
                                pressure=pressure,
                                time_sec=time_sec,
                                gamma=gamma,
                                V_disp=V_disp,
                                stroke=stroke,
                                con_rod=con_rod,
                                compression_ratio=compression_ratio,
                                V_clearance=V_clearance,
                                is_diesel=is_diesel,
                                soi=start_of_injection if is_diesel else None,
                                injection_duration=injection_duration if is_diesel else None,
                                knock_threshold=knock_threshold
                            )
                            fig_dp_all.add_trace(go.Scatter(
                                x=theta,
                                y=results['dp_dtheta'],
                                mode='lines',
                                name=f'Cycle {c}',
                                hovertemplate='Angle: %{x}°<br>dP/dθ: %{y:.4f} bar/deg<extra></extra>'
                            ))
                    
                    fig_dp_all.update_layout(
                        title="dP/dθ vs Crank Angle (All Cycles)",
                        xaxis_title="Crank Angle (deg)",
                        yaxis_title="dP/dθ (bar/deg)",
                        hovermode="x unified",
                        showlegend=True
                    )
                    st.plotly_chart(fig_dp_all, use_container_width=True)
                
                with mtabs[3]:  # HRR (All Cycles)
                    fig_hrr_all = go.Figure()
                    for c in sorted(df['cycle'].dropna().unique()):
                        df_c = df[df['cycle'] == c]
                        theta = df_c['crank_angle'].to_numpy()
                        pressure = df_c['pressure'].to_numpy()
                        time_sec = df_c['time_sec'].to_numpy()
                        
                        if len(theta) >= 2:
                            results = calculate_combustion_metrics(
                                theta=theta,
                                pressure=pressure,
                                time_sec=time_sec,
                                gamma=gamma,
                                V_disp=V_disp,
                                stroke=stroke,
                                con_rod=con_rod,
                                compression_ratio=compression_ratio,
                                V_clearance=V_clearance,
                                is_diesel=is_diesel,
                                soi=start_of_injection if is_diesel else None,
                                injection_duration=injection_duration if is_diesel else None,
                                knock_threshold=knock_threshold
                            )
                            fig_hrr_all.add_trace(go.Scatter(
                                x=theta,
                                y=results['HRR'],
                                mode='lines',
                                name=f'Cycle {c}',
                                hovertemplate='Angle: %{x}°<br>HRR: %{y:.2f} J/deg<extra></extra>'
                            ))
                    
                    fig_hrr_all.update_layout(
                        title="Heat Release Rate (All Cycles)",
                        xaxis_title="Crank Angle (deg)",
                        yaxis_title="HRR (J/deg)",
                        hovermode="x unified",
                        showlegend=True
                    )
                    st.plotly_chart(fig_hrr_all, use_container_width=True)
                
                with mtabs[4]:  # PV Diagrams (All Cycles)
                    fig_pv_all = go.Figure()
                    for c in sorted(df['cycle'].dropna().unique()):
                        df_c = df[df['cycle'] == c]
                        theta = df_c['crank_angle'].to_numpy()
                        pressure = df_c['pressure'].to_numpy()
                        time_sec = df_c['time_sec'].to_numpy()
                        
                        if len(theta) >= 2:
                            results = calculate_combustion_metrics(
                                theta=theta,
                                pressure=pressure,
                                time_sec=time_sec,
                                gamma=gamma,
                                V_disp=V_disp,
                                stroke=stroke,
                                con_rod=con_rod,
                                compression_ratio=compression_ratio,
                                V_clearance=V_clearance,
                                is_diesel=is_diesel,
                                soi=start_of_injection if is_diesel else None,
                                injection_duration=injection_duration if is_diesel else None,
                                knock_threshold=knock_threshold
                            )
                            fig_pv_all.add_trace(go.Scatter(
                                x=results['volume'] * 1e6,
                                y=results['pressure'],
                                mode='lines',
                                name=f'Cycle {c}',
                                hovertemplate='Volume: %{x:.2f} cc<br>Pressure: %{y:.2f} bar<extra></extra>'
                            ))
                    
                    fig_pv_all.update_layout(
                        title="PV Diagrams (All Cycles)",
                        xaxis_title="Volume (cc)",
                        yaxis_title="Pressure (bar)",
                        hovermode="closest",
                        showlegend=True
                    )
                    st.plotly_chart(fig_pv_all, use_container_width=True)
                
                # Multi-Cycle Metrics Summary
                st.subheader("📌 Multi-Cycle Metrics Summary")
                
                cycle_metrics = []
                for c in sorted(df['cycle'].dropna().unique()):
                    df_c = df[df['cycle'] == c]
                    theta = df_c['crank_angle'].to_numpy()
                    pressure = df_c['pressure'].to_numpy()
                    time_sec = df_c['time_sec'].to_numpy()
                    
                    if len(theta) >= 2:
                        results = calculate_combustion_metrics(
                            theta=theta,
                            pressure=pressure,
                            time_sec=time_sec,
                            gamma=gamma,
                            V_disp=V_disp,
                            stroke=stroke,
                            con_rod=con_rod,
                            compression_ratio=compression_ratio,
                            V_clearance=V_clearance,
                            is_diesel=is_diesel,
                            soi=start_of_injection if is_diesel else None,
                            injection_duration=injection_duration if is_diesel else None,
                            knock_threshold=knock_threshold
                        )
                        ignition_idx = np.argmin(np.abs(theta - ignition_timing))
                        
                        metrics = {
                            'Cycle': c,
                            'Max Pressure (bar)': np.max(results['pressure']),
                            'CA50 (deg)': results['CA50'],
                            'IMEP (bar)': results['IMEP'],
                            'Peak HRR (J/deg)': np.max(results['HRR']),
                            'Ignition Time (s)': results['time_sec'][ignition_idx] if ignition_idx < len(results['time_sec']) else np.nan,
                            'Knock Detected': 'Yes' if results['knock']['is_knocking'] else 'No',
                            'Knock Intensity': results['knock']['knock_intensity'],
                            'Knock Events': len(results['knock']['knock_angles'])
                        }
                        
                        if is_diesel and results.get('diesel_metrics'):
                            dm = results['diesel_metrics']
                            metrics.update({
                                'Ignition Delay (°)': dm.get('ignition_delay', np.nan),
                                'Combustion Start (°)': dm.get('combustion_start', np.nan),
                                'Premixed Duration (°)': (dm['premixed_end'] - dm['premixed_start']) if not np.isnan(dm.get('premixed_end', np.nan)) else np.nan,
                                'Diffusion Duration (°)': (dm['diffusion_end'] - dm['diffusion_start']) if not np.isnan(dm.get('diffusion_end', np.nan)) else np.nan,
                                'Premixed Ratio (%)': dm.get('premixed_ratio', np.nan)*100 if not np.isnan(dm.get('premixed_ratio', np.nan)) else np.nan
                            })
                        
                        cycle_metrics.append(metrics)
                
                metrics_df = pd.DataFrame(cycle_metrics)
                st.dataframe(metrics_df.round(2))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Please ensure the Excel file has the correct format.")
else:
    st.info("⬆️ Please upload an Excel file to begin analysis.")
