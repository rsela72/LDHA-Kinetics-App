import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress
from scipy.optimize import curve_fit
from io import StringIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="LDHA Kinetics Analyzer", layout="wide")

# --- CONSTANTS & CONFIG ---
st.sidebar.header("🔬 Assay Configuration")
EPSILON = st.sidebar.number_input("Extinction Coefficient (ε) [M⁻¹ cm⁻¹]", value=6220, help="NADH at 340nm is typically 6220")
PATH_LENGTH = st.sidebar.number_input("Cuvette Path Length (cm)", value=1.0)
ENZYME_CONCENTRATION = st.sidebar.number_input("Enzyme Concentration [µM] (optional for Kcat)", value=0.0, format="%.3f", help="Enter purified enzyme concentration in µM for Kcat calculation.")

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

    # Detect AS
    initial_as_idx = 0
    rolling_diff = pd.Series(abs_vals).rolling(window=15).mean().diff()
    for i in range(15, len(rolling_diff) - 20):
        if all(rolling_diff[i:i+10] < 0):
            initial_as_idx = i - 15 + np.argmax(abs_vals[max(0, i-20):i+1])
            break

    # Detect AE
    initial_ae_idx = len(abs_vals) - 1
    for i in range(initial_as_idx + 50, len(abs_vals) - 50):
        window = abs_vals[i:i+50]
        if np.std(window) < 0.0005 or (abs(window[-1] - window[0]) < 0.001):
            initial_ae_idx = i
            break

    reaction_length = initial_ae_idx - initial_as_idx
    ae_adjustment = int(0.20 * reaction_length)
    return initial_as_idx, max(initial_as_idx + 59, initial_ae_idx - ae_adjustment)

def calculate_kinetics(df, filename):
    """Calculates V0 using the 5-chunk minimum variance method."""
    as_idx, ae_idx = detect_as_ae(df)
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

    if len(chunks) < 5:
        final_reg = linregress(subset['Time'], subset['Abs'])
        v0_data = subset
    else:
        variances = [np.var([c['slope'] for c in chunks[j:j+5]]) for j in range(len(chunks)-4)]
        best_start = np.argmin(variances)
        v0_data = subset.iloc[chunks[best_start]['start'] : chunks[best_start+4]['end']]
        final_reg = linregress(v0_data['Time'], v0_data['Abs'])

    # Units: (ΔAbs/min) -> (M/min) -> (µM/s)
    # V = (Slope / (ε * l)) * 10^6 / 60
    abs_slope = abs(final_reg.slope)
    v0_um_s = (abs_slope / (EPSILON * PATH_LENGTH)) * 1e6 / 60

    pyr_match = re.search(r'(\d+[,.]?\d*)\s*mM', filename, re.IGNORECASE)
    pyr_val = float(pyr_match.group(1).replace(',', '.')) if pyr_match else 0.0

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
        "full_df": df
    }

def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

# --- UI APP START ---

st.title("🧪 LDHA Michaelis-Menten Analyzer")
st.markdown("""
Upload your kinetic CSV files. The app will automatically calculate the initial velocity ($V_0$)
from the steadiest part of each curve and fit the data to the Michaelis-Menten equation.
""")

files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)

if files:
    all_runs = []
    for f in files:
        df = load_and_clean_csv(f)
        if df is not None:
            res = calculate_kinetics(df, f.name)
            all_runs.append(res)

    if all_runs:
        # 1. OUTLIER SELECTION TABLE
        st.subheader("📊 Results Summary & Outlier Selection")
        st.info("Uncheck the 'Include' box to exclude a run from the Michaelis-Menten fit.")

        # Build dataframe for editor
        summary_data = []
        for r in all_runs:
            summary_data.append({
                "Include": True,
                "File": r['filename'],
                "Pyruvate (mM)": r['pyruvate'],
                "V0 (µM/s)": round(r['v0_um_s'], 4),
                "R²": round(r['r2'], 4)
            })

        edited_df = st.data_editor(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        # Filter included runs
        included_filenames = edited_df[edited_df["Include"] == True]["File"].tolist()
        final_results = [r for r in all_runs if r['filename'] in included_filenames]

        # 2. CALCULATIONS BREAKDOWN
        with st.expander("📝 View Detailed Calculations"):
            st.latex(r"V_0 (\mu M/s) = \frac{|\Delta Abs/min|}{\epsilon \cdot l} \cdot \frac{10^6}{60}")
            calc_table = []
            for r in final_results:
                calc_table.append({
                    "Substrate [S]": f"{r['pyruvate']} mM",
                    "Slope (Abs/min)": round(r['slope_abs_min'], 6),
                    "Step 1: (Slope / ε)": f"{r['slope_abs_min']/EPSILON:.2e} M/min",
                    "Final V0": f"{r['v0_um_s']:.4f} µM/s"
                })
            st.table(calc_table)

        # 3. INDIVIDUAL V0 PLOTS
        st.divider()
        st.subheader("📈 Individual Run Fits")
        cols = st.columns(2)
        for idx, r in enumerate(all_runs):
            with cols[idx % 2].expander(f"Run: {r['pyruvate']} mM Pyruvate ({r['filename']})", expanded=False):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
                # Plot 1: Full
                ax1.plot(r['full_df']['Time'], r['full_df']['Abs'], color='gray', alpha=0.5)
                ax1.axvline(r['as_time'], color='g', linestyle='--', label='Start')
                ax1.axvline(r['ae_time'], color='r', linestyle='--', label='End')
                ax1.set_title("Full Assay")
                # Plot 2: Fit
                ax2.scatter(r['v0_data']['Time'], r['v0_data']['Abs'], s=5, color='orange')
                t = r['v0_data']['Time']
                ax2.plot(t, r['slope_abs_min'] * t + r['intercept'], color='blue', lw=1)
                ax2.set_title(f"V0 Slope (R²={r['r2']:.4f})")
                st.pyplot(fig)

        # 4. MICHAELIS-MENTEN FIT
        if len(final_results) >= 3:
            st.divider()
            st.subheader("🧪 Michaelis-Menten Kinetic Fit")

            s_vals = np.array([r['pyruvate'] for r in final_results])
            v_vals = np.array([r['v0_um_s'] for r in final_results])

            try:
                popt, pcov = curve_fit(michaelis_menten, s_vals, v_vals, p0=[max(v_vals), 1.0])
                vmax, km = popt
                perr = np.sqrt(np.diag(pcov))

                vmax_err = perr[0]
                km_err = perr[1]

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Km (Michaelis Constant)", f"{km:.3f} ± {km_err:.3f} mM")
                col_res2.metric("Vmax (Max Velocity)", f"{vmax:.3f} ± {vmax_err:.3f} µM/s")

                if ENZYME_CONCENTRATION > 0:
                    kcat = vmax / ENZYME_CONCENTRATION
                    # Error propagation for Kcat = Vmax / [E]
                    # Assuming enzyme concentration has negligible error compared to Vmax
                    kcat_err = kcat * (vmax_err / vmax)
                    col_res3.metric("Kcat (Turnover Number)", f"{kcat:.3f} ± {kcat_err:.3f} s⁻¹")
                else:
                    col_res3.metric("Kcat (Turnover Number)", "N/A (Enter enzyme conc.)")

                fig_mm, ax_mm = plt.subplots(figsize=(8, 5))
                s_plot = np.linspace(0, max(s_vals)*1.2, 100)
                ax_mm.scatter(s_vals, v_vals, color='red', label='Experimental Data', zorder=5)
                ax_mm.plot(s_plot, michaelis_menten(s_plot, *popt), label=f'MM Fit ($K_m$={km:.2f})', color='black')
                ax_mm.set_xlabel("Pyruvate Concentration [mM]")
                ax_mm.set_ylabel("Initial Velocity $V_0$ [µM/s]")
                ax_mm.legend()
                ax_mm.grid(True, which='both', linestyle='--', alpha=0.5)
                st.pyplot(fig_mm)

            except Exception as e:
                st.error(f"Could not fit Michaelis-Menten curve. Ensure you have enough data points. Error: {e}")
        else:
            st.warning("Please include at least 3 runs to perform Michaelis-Menten fitting.")

else:
    st.write("---")
    st.info("☝️ Please upload some CSV files to begin.")