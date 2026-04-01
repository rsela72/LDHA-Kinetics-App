import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress
from scipy.optimize import curve_fit
from io import StringIO
import matplotlib.ticker as mticker

# --- PAGE CONFIG ---
st.set_page_config(page_title="LDHA Kinetics Analyzer", layout="wide")

# --- CONSTANTS & CONFIG ---
st.sidebar.header("🔬 Assay Configuration")
EPSILON = st.sidebar.number_input("Extinction Coefficient (ε) [M⁻¹ cm⁻¹]", value=6220, help="NADH at 340nm is typically 6220")
PATH_LENGTH = st.sidebar.number_input("Cuvette Path Length (cm)", value=1.0)
ENZYME_CONCENTRATION = st.sidebar.number_input("Enzyme Concentration [µM] (optional for Kcat)", value=0.0, format="%.3f", help="Enter purified enzyme concentration in µM for Kcat calculation.")
NUM_CHUNKS_FOR_VARIANCE = st.sidebar.number_input("Number of continuous chunks for V0 detection", value=8, min_value=3, help="Increase to make V0 detection more stringent over a longer linear phase. Default is 8.")

# --- CORE LOGIC FUNCTIONS (Refactored from Notebook) ---

def load_and_clean_csv(uploaded_file):
    """Parses Streamlit's UploadedFile object into a clean DataFrame."""
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.splitlines()

        start_idx = -1
        for i, line in enumerate(lines):
            if "Time (min)" in line:
                start_idx = i
                break
        if start_idx == -1: return None

        data_lines = [lines[start_idx]]
        for line in lines[start_idx+1:]:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    data_lines.append(line)
                except ValueError: break

        df = pd.read_csv(StringIO("\n".join(data_lines)))
        df = df.iloc[:, [0, 1]]
        df.columns = ['Time', 'Abs']
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

def detect_as_ae(df):
    """Identifies the start (AS) and plateau (AE) of the reaction."""
    if len(df) < 60: return 0, len(df) - 1
    abs_vals = df['Abs'].values

    # Detect AS based on absorbance threshold of 0.47 (MODIFIED LOGIC)
    threshold_indices = df[df['Abs'] <= 0.47].index
    if not threshold_indices.empty:
        initial_as_idx = threshold_indices[0]
    else:
        initial_as_idx = 0 # Fallback if threshold is never reached

    # Detect AE (unchanged logic)
    initial_ae_idx = len(abs_vals) - 1
    for i in range(initial_as_idx + 50, len(abs_vals) - 50):
        window = abs_vals[i:i+50]
        if np.std(window) < 0.0005 or (abs(window[-1] - window[0]) < 0.001):
            initial_ae_idx = i
            break

    reaction_length = initial_ae_idx - initial_as_idx
    ae_adjustment = int(0.20 * reaction_length)
    return initial_as_idx, max(initial_as_idx + 59, initial_ae_idx - ae_adjustment)

def calculate_kinetics(df, filename, user_as_time=None, user_ae_time=None):
    """Calculates V0 using the 5-chunk minimum variance method, with optional user-defined AS/AE times."""
    if user_as_time is not None and user_ae_time is not None:
        # Find closest indices for user-defined times
        as_idx = df['Time'].sub(user_as_time).abs().idxmin()
        ae_idx = df['Time'].sub(user_ae_time).abs().idxmin()
    else:
        as_idx, ae_idx = detect_as_ae(df)

    # Ensure as_idx is before ae_idx and within bounds
    if as_idx >= ae_idx: # If user input causes this, fallback to automatic detection
        as_idx, ae_idx = detect_as_ae(df)
    elif as_idx < 0: as_idx = 0
    elif ae_idx >= len(df): ae_idx = len(df) - 1

    subset = df.iloc[as_idx:ae_idx+1].copy().reset_index(drop=True)
    N = len(subset)

    chunk_size = max(15, int(np.round(0.025 * N)))
    step_size = max(5, int(np.round(0.020 * N)))

    chunks = []
    i = 0
    while i + chunk_size <= N:
        chunk_df = subset.iloc[i : i + chunk_size]
        reg = linregress(chunk_df['Time'], chunk_df['Abs'])
        chunks.append({'slope': reg.slope, 'start': i, 'end': i + chunk_size, 'r_val': reg.rvalue})
        i += step_size

    if len(chunks) < NUM_CHUNKS_FOR_VARIANCE:
        final_reg = linregress(subset['Time'], subset['Abs'])
        v0_data = subset
    else:
        # Use NUM_CHUNKS_FOR_VARIANCE here
        variances = [np.var([c['slope'] for c in chunks[j:j+int(NUM_CHUNKS_FOR_VARIANCE)]]) for j in range(len(chunks)-int(NUM_CHUNKS_FOR_VARIANCE)+1)]
        best_start = np.argmin(variances)
        v0_data = subset.iloc[chunks[best_start]['start'] : chunks[best_start+int(NUM_CHUNKS_FOR_VARIANCE)-1]['end']]
        final_reg = linregress(v0_data['Time'], v0_data['Abs'])

    # Units: (ΔAbs/min) -> (M/min) -> (µM/s)
    # V = (Slope / (ε * l)) * 10^6 / 60
    abs_slope = abs(final_reg.slope)
    v0_um_s = (abs_slope / (EPSILON * PATH_LENGTH)) * 1e6 / 60

    # Extract pyruvate concentration
    pyr_match = re.search(r'(\d+[,.]?\d*)\s*mM', filename, re.IGNORECASE)
    pyr_val = float(pyr_match.group(1).replace(',', '.')) if pyr_match else 0.0

    # Extract enzyme type based on "initials_enzymetype_concentration_runorder_date" convention
    parts = filename.split('_')
    enzyme_type = "Unknown"
    if len(parts) > 1:
        # Assuming enzyme type is the second part (index 1)
        enzyme_type = parts[1].split('.')[0] # Remove file extension if present

    # Convert to uppercase for consistency
    enzyme_type = enzyme_type.upper()

    return {
        "filename": filename,
        "pyruvate": pyr_val,
        "v0_um_s": v0_um_s,
        "slope_abs_min": final_reg.slope,
        "intercept": final_reg.intercept,
        "r2": final_reg.rvalue**2,
        "as_time": df.iloc[as_idx]['Time'],
        "ae_time": df.iloc[ae_idx]['Time'],
        "v0_data": v0_data,
        "full_df": df,
        "enzyme_type": enzyme_type # Add enzyme type here
    }

def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

# --- UI APP START ---

st.title("🧪 LDHA Michaelis-Menten Analyzer")
st.markdown("""
Upload your kinetic CSV files. The app will automatically calculate the initial velocity ($V_0$)
from the steadiest part of each curve and fit the data to the Michaelis-Menten equation.
""")

# Initialize session state for all_runs and uploaded file names
if "all_runs" not in st.session_state:
    st.session_state.all_runs = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
# New session state variables for the two-step process
if "v0_calculated_results" not in st.session_state:
    st.session_state.v0_calculated_results = []
if "v0_calculation_done" not in st.session_state:
    st.session_state.v0_calculation_done = False
if "mm_plot_ready" not in st.session_state:
    st.session_state.mm_plot_ready = False # New state variable to control MM plot display

# Initialize MM Plot Customization session state variables
if "mm_plot_title" not in st.session_state:
    st.session_state.mm_plot_title = "Michaelis-Menten Fit"
if "mm_x_axis_label" not in st.session_state:
    st.session_state.mm_x_axis_label = "Pyruvate Concentration [mM]"
if "mm_y_axis_label" not in st.session_state:
    st.session_state.mm_y_axis_label = "Initial Velocity $V_0$ [µM/s]"
if "mm_title_font_size" not in st.session_state:
    st.session_state.mm_title_font_size = 14
if "mm_label_font_size" not in st.session_state:
    st.session_state.mm_label_font_size = 12
if "mm_tick_font_size" not in st.session_state:
    st.session_state.mm_tick_font_size = 10
if "mm_legend_font_size" not in st.session_state:
    st.session_state.mm_legend_font_size = 10
if "mm_data_point_color" not in st.session_state:
    st.session_state.mm_data_point_color = "#FF0000"
if "mm_fit_line_color" not in st.session_state:
    st.session_state.mm_fit_line_color = "#000000"
if "mm_show_grid" not in st.session_state:
    st.session_state.mm_show_grid = True
if "mm_font_family" not in st.session_state:
    st.session_state.mm_font_family = "sans-serif"
if "mm_decimal_places" not in st.session_state:
    st.session_state.mm_decimal_places = 3
if "mm_plot_width" not in st.session_state:
    st.session_state.mm_plot_width = 6.8 # Default width
if "mm_plot_height" not in st.session_state:
    st.session_state.mm_plot_height = 6.5 # Default height
if "mm_axis_decimal_places" not in st.session_state:
    st.session_state.mm_axis_decimal_places = 2 # Default for axis ticks

files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)

if files:
    current_file_names = [f.name for f in files]

    # Check if the set of uploaded files has changed
    if set(current_file_names) != set(st.session_state.uploaded_file_names):
        st.session_state.all_runs = [] # Clear previous runs if new files are detected
        st.session_state.v0_calculated_results = [] # Clear V0 results as files changed
        st.session_state.v0_calculation_done = False
        st.session_state.mm_plot_ready = False # Reset MM plot ready state
        st.session_state.uploaded_file_names = current_file_names

        for f in files:
            df = load_and_clean_csv(f)
            if df is not None:
                res = calculate_kinetics(df, f.name)
                st.session_state.all_runs.append(res)

    # Use the all_runs from session state for all subsequent operations
    all_runs_from_session = st.session_state.all_runs

    if all_runs_from_session:
        # 1. OUTLIER SELECTION TABLE
        st.subheader("📊 Results Summary & Outlier Selection")
        st.info("Uncheck the 'Include' box to exclude a run from the Michaelis-Menten fit. You can also adjust Pyruvate Concentration, Enzyme Type, and AS/AE times here.")

        # Build dataframe for editor using current session state
        summary_data = []
        for r in all_runs_from_session:
            summary_data.append({
                "Include": True,
                "File": r['filename'],
                "Pyruvate (mM)": round(r['pyruvate'], 4),
                "V0 (µM/s)": round(r['v0_um_s'], 4),
                "R²": round(r['r2'], 4),
                "Enzyme Type": r['enzyme_type'], # Add enzyme type to summary_data
                "AS (min)": round(r['as_time'], 3), # Add AS here for editing
                "AE (min)": round(r['ae_time'], 3)  # Add AE here for editing
            })

        edited_df = st.data_editor(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Pyruvate (mM)": st.column_config.NumberColumn(
                    "Pyruvate (mM)",
                    format="%.3f",
                    disabled=False
                ),
                "Enzyme Type": st.column_config.TextColumn(
                    "Enzyme Type",
                    disabled=False
                ),
                "AS (min)": st.column_config.NumberColumn(
                    "AS (min)",
                    format="%.3f",
                    disabled=False
                ),
                "AE (min)": st.column_config.NumberColumn(
                    "AE (min)",
                    format="%.3f",
                    disabled=False
                )
            }
        )

        # Update pyruvate concentrations, enzyme types, AS, and AE based on edited_df for *all* runs in session state
        for i, row in edited_df.iterrows():
            original_filename = row['File']
            for run_state in st.session_state.all_runs:
                if run_state['filename'] == original_filename:
                    if run_state['pyruvate'] != row['Pyruvate (mM)']:
                        run_state['pyruvate'] = row['Pyruvate (mM)']
                    if run_state['enzyme_type'] != row['Enzyme Type']:
                        run_state['enzyme_type'] = row['Enzyme Type']
                    # Update AS and AE times in session state. V0 will be recalculated on button press.
                    if run_state['as_time'] != row['AS (min)']:
                        run_state['as_time'] = row['AS (min)']
                    if run_state['ae_time'] != row['AE (min)']:
                        run_state['ae_time'] = row['AE (min)']
                    break

        # Filter included filenames after updating state based on edited_df
        included_filenames = edited_df[edited_df["Include"] == True]["File"].tolist()

        # Button to trigger V0 calculation
        if st.button("Calculate V0"):
            st.session_state.v0_calculated_results = [] # Clear previous V0 results
            for idx, run_state in enumerate(st.session_state.all_runs):
                if run_state['filename'] in included_filenames:
                    recalculated_res = calculate_kinetics(
                        run_state['full_df'],
                        run_state['filename'],
                        user_as_time=run_state['as_time'],
                        user_ae_time=run_state['ae_time']
                    )
                    # Update the run_state object in session_state with new V0, slope, r2, etc.
                    st.session_state.all_runs[idx].update({
                        "v0_um_s": recalculated_res["v0_um_s"],
                        "slope_abs_min": recalculated_res["slope_abs_min"],
                        "intercept": recalculated_res["intercept"],
                        "r2": recalculated_res["r2"],
                        "v0_data": recalculated_res["v0_data"]
                    })
                    st.session_state.v0_calculated_results.append(st.session_state.all_runs[idx])
            st.session_state.v0_calculation_done = True
            st.session_state.mm_plot_ready = False # Reset MM plot ready state if V0s are recalculated
            st.success("V0 calculations complete!")

        # Display Calculations Breakdown and Individual Run Fits if V0s are calculated
        if st.session_state.v0_calculation_done and st.session_state.v0_calculated_results:
            final_results = st.session_state.v0_calculated_results # Data for breakdown and individual plots

            # 2. CALCULATIONS BREAKDOWN
            with st.expander("📝 View Detailed Calculations"):
                st.latex(r"V_0 (\mu M/s) = \frac{|\Delta Abs/min|}{\epsilon \cdot l} \cdot \frac{10^6}{60}")
                calc_table = []
                for r in final_results:
                    calc_table.append({
                        "Substrate [S]": f"{r['pyruvate']} mM",
                        "Slope (Abs/min)": round(r['slope_abs_min'], 6),
                        "Step 1: (Slope / \u03b5)": f"{r['slope_abs_min']/EPSILON:.2e} M/min",
                        "Final V0": f"{r['v0_um_s']:.4f} \u00b5M/s"
                    })
                st.table(calc_table)

            # 3. INDIVIDUAL V0 PLOTS
            st.divider()
            st.subheader("📈 Individual Run Fits")
            cols = st.columns(2)
            for idx, r_to_plot in enumerate(st.session_state.all_runs): # Iterate through session_state.all_runs for plotting
                # Ensure only included files with recalculated V0 are plotted or handle others gracefully
                if r_to_plot['filename'] in included_filenames: # Use included_filenames from data_editor
                    with cols[idx % 2].expander(f"Run: {r_to_plot['pyruvate']} mM Pyruvate ({r_to_plot['filename']})", expanded=False):

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
                        # Plot 1: Full
                        ax1.plot(r_to_plot['full_df']['Time'], r_to_plot['full_df']['Abs'], color='gray', alpha=0.5)
                        ax1.axvline(r_to_plot['as_time'], color='g', linestyle='--', label='Start')
                        ax1.axvline(r_to_plot['ae_time'], color='r', linestyle='--', label='End')
                        ax1.set_title("Full Assay")
                        # Plot 2: Fit
                        ax2.scatter(r_to_plot['v0_data']['Time'], r_to_plot['v0_data']['Abs'], s=5, color='orange')
                        t = r_to_plot['v0_data']['Time']
                        ax2.plot(t, r_to_plot['slope_abs_min'] * t + r_to_plot['intercept'], color='blue', lw=1)

                        # Individual V0 Plot text
                        v0_plot_text = (
                            f"R² = {r_to_plot['r2']:.4f}\n"
                            f"V0 = {r_to_plot['v0_um_s']:.4f} \u00b5M/s\n"
                            f"[S] = {r_to_plot['pyruvate']} mM\n"
                            f"Enzyme: {r_to_plot['enzyme_type']}"
                        )
                        ax2.text(0.95, 0.95, v0_plot_text, transform=ax2.transAxes, fontsize=8,
                                 verticalalignment='top', horizontalalignment='right',
                                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
                        ax2.set_title("V0 Fit")
                        st.pyplot(fig)

            # Button to initiate MM Plotting
            if st.button("Initiate MM Plot"):
                st.session_state.mm_plot_ready = True

            # 4. MICHAELIS-MENTEN FIT (now conditional on mm_plot_ready)
            if st.session_state.mm_plot_ready:
                final_results_mm_fit = st.session_state.v0_calculated_results # Data for MM fit

                if len(final_results_mm_fit) >= 3:
                    st.divider()
                    st.subheader("🧪 Michaelis-Menten Kinetic Fit")

                    s_vals = np.array([r['pyruvate'] for r in final_results_mm_fit])
                    v_vals = np.array([r['v0_um_s'] for r in final_results_mm_fit])

                    # Get unique enzyme types for the MM plot text
                    unique_enzyme_types = sorted(list(set([r['enzyme_type'] for r in final_results_mm_fit])))
                    enzyme_types_str = "" if not unique_enzyme_types else f"\nEnzyme Type(s): {', '.join(unique_enzyme_types)}"

                    # --- Plot Customization Options ---
                    with st.sidebar.expander("📈 Michaelis-Menten Plot Customization", expanded=True):
                        st.session_state.mm_plot_title = st.text_input(
                            "Plot Title",
                            value=st.session_state.mm_plot_title,
                            key="mm_plot_title_widget"
                        )
                        st.session_state.mm_x_axis_label = st.text_input(
                            "X-axis Label",
                            value=st.session_state.mm_x_axis_label,
                            key="mm_x_axis_label_widget"
                        )
                        st.session_state.mm_y_axis_label = st.text_input(
                            "Y-axis Label",
                            value=st.session_state.mm_y_axis_label,
                            key="mm_y_axis_label_widget"
                        )
                        st.session_state.mm_title_font_size = st.number_input(
                            "Title Font Size",
                            value=st.session_state.mm_title_font_size,
                            min_value=8,
                            key="mm_title_font_size_widget"
                        )
                        st.session_state.mm_label_font_size = st.number_input(
                            "Label Font Size",
                            value=st.session_state.mm_label_font_size,
                            min_value=8,
                            key="mm_label_font_size_widget"
                        )
                        st.session_state.mm_tick_font_size = st.number_input(
                            "Tick Font Size",
                            value=st.session_state.mm_tick_font_size,
                            min_value=6,
                            key="mm_tick_font_size_widget"
                        )
                        st.session_state.mm_legend_font_size = st.number_input(
                            "Legend Font Size",
                            value=st.session_state.mm_legend_font_size,
                            min_value=6,
                            key="mm_legend_font_size_widget"
                        )
                        st.session_state.mm_data_point_color = st.color_picker(
                            "Experimental Data Color",
                            value=st.session_state.mm_data_point_color,
                            key="mm_data_point_color_widget"
                        )
                        st.session_state.mm_fit_line_color = st.color_picker(
                            "Fit Line Color",
                            value=st.session_state.mm_fit_line_color,
                            key="mm_fit_line_color_widget"
                        )
                        st.session_state.mm_show_grid = st.checkbox(
                            "Show Grid",
                            value=st.session_state.mm_show_grid,
                            key="mm_show_grid_widget"
                        )
                        st.session_state.mm_font_family = st.selectbox(
                            "Font Family",
                            options=["Times New Roman", "sans-serif", "serif", "monospace", "cursive", "fantasy"],
                            index=1, # Default to sans-serif
                            key="mm_font_family_widget"
                        )
                        st.session_state.mm_decimal_places = st.number_input(
                            "Number of Decimal Places for Kinetics Values",
                            value=st.session_state.mm_decimal_places,
                            min_value=0, max_value=10,
                            key="mm_decimal_places_widget"
                        )
                        st.session_state.mm_axis_decimal_places = st.number_input(
                            "Number of Decimal Places for Axis Ticks",
                            value=st.session_state.mm_axis_decimal_places,
                            min_value=0, max_value=10,
                            key="mm_axis_decimal_places_widget"
                        )
                        st.session_state.mm_plot_width = st.number_input(
                            "Plot Width (inches)",
                            value=st.session_state.mm_plot_width,
                            min_value=1.0, max_value=20.0,
                            key="mm_plot_width_widget"
                        )
                        st.session_state.mm_plot_height = st.number_input(
                            "Plot Height (inches)",
                            value=st.session_state.mm_plot_height,
                            min_value=1.0, max_value=20.0,
                            key="mm_plot_height_widget"
                        )
                    # --- End Plot Customization Options ---

                    try:
                        popt, pcov = curve_fit(michaelis_menten, s_vals, v_vals, p0=[max(v_vals), 1.0])
                        vmax, km = popt
                        perr = np.sqrt(np.diag(pcov))

                        vmax_err = perr[0]
                        km_err = perr[1]

                        f_str = f".{st.session_state.mm_decimal_places}f"
                        axis_f_str = f"%.{st.session_state.mm_axis_decimal_places}f"

                        col_res1, col_res2, col_res3 = st.columns(3)
                        col_res1.metric("Km (Michaelis Constant)", f"{km:{f_str}} \u00b1 {km_err:{f_str}} mM")
                        col_res2.metric("Vmax (Max Velocity)", f"{vmax:{f_str}} \u00b1 {vmax_err:{f_str}} \u00b5M/s")

                        # Build comprehensive label for the MM Fit line
                        fit_label_parts = [
                            f'MM Fit ($K_m$={km:{f_str}} mM)',
                            f'Vmax={vmax:{f_str}} \u00b5M/s'
                        ]
                        if ENZYME_CONCENTRATION > 0:
                            kcat = vmax / ENZYME_CONCENTRATION
                            # Error propagation for Kcat = Vmax / [E]
                            # Assuming enzyme concentration has negligible error compared to Vmax
                            kcat_err = kcat * (vmax_err / vmax)
                            col_res3.metric("Kcat (Turnover Number)", f"{kcat:{f_str}} \u00b1 {kcat_err:{f_str}} s\u207b\u00b9")
                            fit_label_parts.append(f'Kcat={kcat:{f_str}} s\u207b\u00b9')
                        else:
                            col_res3.metric("Kcat (Turnover Number)", "N/A (Enter enzyme conc.)")

                        if unique_enzyme_types:
                            fit_label_parts.append(f'Enzyme Type(s): {', '.join(unique_enzyme_types)}')

                        # Use '\n' for multi-line legend entry
                        fit_label = '\n'.join(fit_label_parts)

                        plt.rcParams['font.family'] = st.session_state.mm_font_family
                        fig_mm, ax_mm = plt.subplots(figsize=(st.session_state.mm_plot_width, st.session_state.mm_plot_height))
                        s_plot = np.linspace(0, max(s_vals)*1.2, 100)
                        ax_mm.scatter(s_vals, v_vals, color=st.session_state.mm_data_point_color, label='Experimental Data', zorder=5)
                        ax_mm.plot(s_plot, michaelis_menten(s_plot, *popt), label=fit_label, color=st.session_state.mm_fit_line_color)

                        # Apply customization options
                        ax_mm.set_xlabel(st.session_state.mm_x_axis_label, fontsize=st.session_state.mm_label_font_size)
                        ax_mm.set_ylabel(st.session_state.mm_y_axis_label, fontsize=st.session_state.mm_label_font_size)
                        ax_mm.set_title(st.session_state.mm_plot_title, fontsize=st.session_state.mm_title_font_size)
                        ax_mm.tick_params(axis='x', labelsize=st.session_state.mm_tick_font_size)
                        ax_mm.tick_params(axis='y', labelsize=st.session_state.mm_tick_font_size)

                        # Apply axis decimal formatting
                        formatter = mticker.FormatStrFormatter(axis_f_str)
                        ax_mm.xaxis.set_major_formatter(formatter)
                        ax_mm.yaxis.set_major_formatter(formatter)

                        handles, labels = ax_mm.get_legend_handles_labels()
                        ax_mm.legend(handles=handles, labels=labels,
                                     fontsize=st.session_state.mm_legend_font_size,
                                     bbox_to_anchor=(0.5, -0.35), loc='upper center',
                                     borderaxespad=0., ncol=1)
                        if st.session_state.mm_show_grid:
                            ax_mm.grid(True, which='both', linestyle='--', alpha=0.5)

                        # Adjust rect to make more space for legend at bottom.
                        fig_mm.tight_layout(rect=[0, 0.2, 1, 1])
                        st.pyplot(fig_mm)

                    except Exception as e:
                        st.error(f"Could not fit Michaelis-Menten curve. Ensure you have enough data points. Error: {e}")
                        print(f"Michaelis-Menten curve fit error: {e}") # Print the exception for debugging
                else:
                    st.warning("Please include at least 3 runs to perform Michaelis-Menten fitting.")

else:
    # If no files are uploaded, clear session state related to runs
    if 'all_runs' in st.session_state:
        del st.session_state.all_runs
    if 'uploaded_file_names' in st.session_state:
        del st.session_state.uploaded_file_names
    if 'v0_calculated_results' in st.session_state:
        del st.session_state.v0_calculated_results
    if 'v0_calculation_done' in st.session_state:
        del st.session_state.v0_calculation_done
    if 'mm_plot_ready' in st.session_state:
        del st.session_state.mm_plot_ready # Clear mm_plot_ready when no files are loaded

    # The MM plot customization settings will NOT be cleared here to ensure persistence.

    st.write("---")
    st.info("☝️ Please upload some CSV files to begin.")

st.markdown("<p style='font-size: small;'>Roee Sela '27</p>", unsafe_allow_html=True)