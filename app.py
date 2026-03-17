
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress
from scipy.optimize import curve_fit, least_squares
from io import StringIO

APP_VERSION = "v.2"
APP_NAME = f"LDHA Michaelis-Menten Analyzer {APP_VERSION}"

# --- PAGE CONFIG ---
st.set_page_config(page_title=APP_NAME, layout="wide")

# --- CONSTANTS & CONFIG ---
st.sidebar.header("🔬 Assay Configuration")
EPSILON = st.sidebar.number_input(
    "Extinction Coefficient (ε) [M⁻¹ cm⁻¹]",
    value=6220.0,
    help="NADH at 340 nm is typically 6220"
)
PATH_LENGTH = st.sidebar.number_input("Cuvette Path Length (cm)", value=1.0)
ENZYME_CONCENTRATION = st.sidebar.number_input(
    "Enzyme Concentration [µM] (optional for Kcat)",
    value=0.0,
    format="%.3f",
    help="Enter purified enzyme concentration in µM for Kcat calculation."
)

st.sidebar.subheader("Linear-region detection")
MIN_TIME_TO_CONSIDER = st.sidebar.number_input(
    "Ignore data before (min)",
    value=0.33,
    format="%.3f",
    help="Data before this time is ignored when searching for the linear portion."
)
MIN_POINTS_BEFORE_BREAK = st.sidebar.number_input(
    "Min points before breakpoint",
    min_value=3,
    value=6,
    step=1
)
MIN_POINTS_AFTER_BREAK = st.sidebar.number_input(
    "Min points after breakpoint",
    min_value=5,
    value=8,
    step=1
)
MIN_R2_LINEAR = st.sidebar.number_input(
    "Minimum R² for accepted linear window",
    min_value=0.0,
    max_value=1.0,
    value=0.995,
    format="%.4f"
)
WINDOW_POINTS = st.sidebar.number_input(
    "Linear window size (points)",
    min_value=5,
    value=12,
    step=1,
    help="Number of points used when testing for the earliest acceptable linear region after the breakpoint."
)
SLOPE_RELAXATION_FACTOR = st.sidebar.number_input(
    "Max allowed post-break slope / pre-break slope magnitude",
    min_value=0.01,
    max_value=1.0,
    value=0.70,
    format="%.2f",
    help="Post-breakpoint slope magnitude must be smaller than the initial steep slope by at least this factor."
)

def load_and_clean_csv(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.splitlines()

        start_idx = -1
        for i, line in enumerate(lines):
            if "Time (min)" in line:
                start_idx = i
                break
        if start_idx == -1:
            return None

        data_lines = [lines[start_idx]]
        for line in lines[start_idx + 1:]:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    data_lines.append(line)
                except ValueError:
                    break

        df = pd.read_csv(StringIO("\n".join(data_lines)))
        df = df.iloc[:, [0, 1]]
        df.columns = ['Time', 'Abs']
        df = df.dropna().reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

def piecewise_linear_continuous(x, x0, y0, m1, m2):
    return np.where(x < x0, y0 + m1 * (x - x0), y0 + m2 * (x - x0))

def fit_segmented_regression(x, y, min_pts_before=6, min_pts_after=8):
    n = len(x)
    if n < (min_pts_before + min_pts_after + 2):
        return []

    candidates = []

    for bp_idx in range(min_pts_before, n - min_pts_after):
        x0_guess = x[bp_idx]
        y0_guess = y[bp_idx]

        left_reg = linregress(x[:bp_idx], y[:bp_idx])
        right_reg = linregress(x[bp_idx:], y[bp_idx:])

        p0 = [x0_guess, y0_guess, left_reg.slope, right_reg.slope]

        lower = [x[min_pts_before], min(y) - abs(np.ptp(y)), -np.inf, -np.inf]
        upper = [x[n - min_pts_after - 1], max(y) + abs(np.ptp(y)), np.inf, np.inf]

        try:
            res = least_squares(
                lambda p: piecewise_linear_continuous(x, *p) - y,
                x0=p0,
                bounds=(lower, upper),
                max_nfev=5000
            )
            x0, y0, m1, m2 = res.x
            split_idx = np.searchsorted(x, x0)

            if split_idx < min_pts_before or split_idx > n - min_pts_after:
                continue

            if abs(m2) >= abs(m1) * SLOPE_RELAXATION_FACTOR:
                continue

            y_fit = piecewise_linear_continuous(x, x0, y0, m1, m2)
            sse = np.sum((y - y_fit) ** 2)

            candidates.append({
                "x0": x0,
                "y0": y0,
                "m1": m1,
                "m2": m2,
                "split_idx": split_idx,
                "sse": sse
            })

        except Exception:
            continue

    return candidates

def choose_earliest_linear_window_after_break(x, y, start_idx, window_points=12, min_r2=0.995):
    n = len(x)

    for i in range(start_idx, n - window_points + 1):
        xw = x[i:i + window_points]
        yw = y[i:i + window_points]

        reg = linregress(xw, yw)

        if reg.slope >= 0:
            continue
        if reg.rvalue**2 < min_r2:
            continue

        best_j = i + window_points
        best_reg = reg

        for j in range(i + window_points + 1, n + 1):
            xext = x[i:j]
            yext = y[i:j]
            reg_ext = linregress(xext, yext)
            if reg_ext.slope < 0 and reg_ext.rvalue**2 >= min_r2:
                best_j = j
                best_reg = reg_ext
            else:
                break

        return {
            "start_idx": i,
            "end_idx": best_j,
            "reg": best_reg,
            "r2": best_reg.rvalue**2,
            "slope": best_reg.slope
        }

    return None

def calculate_kinetics(df, filename):
    df_fit = df[df["Time"] >= MIN_TIME_TO_CONSIDER].copy().reset_index(drop=True)

    if len(df_fit) < (MIN_POINTS_BEFORE_BREAK + MIN_POINTS_AFTER_BREAK + 2):
        reg = linregress(df_fit["Time"], df_fit["Abs"])
        v0_data = df_fit.copy()
        breakpoint_time = df_fit["Time"].iloc[0]
        segmented_fit = None
        selected_window = None
    else:
        x = df_fit["Time"].to_numpy()
        y = df_fit["Abs"].to_numpy()

        candidates = fit_segmented_regression(
            x, y,
            min_pts_before=MIN_POINTS_BEFORE_BREAK,
            min_pts_after=MIN_POINTS_AFTER_BREAK
        )

        if candidates:
            best_sse = min(c["sse"] for c in candidates)
            near_best = [c for c in candidates if c["sse"] <= best_sse * 1.05]
            segmented_fit = min(near_best, key=lambda c: c["split_idx"])
            breakpoint_idx = segmented_fit["split_idx"]
            breakpoint_time = segmented_fit["x0"]
        else:
            segmented_fit = None
            breakpoint_idx = 0
            breakpoint_time = df_fit["Time"].iloc[0]

        selected_window = choose_earliest_linear_window_after_break(
            x, y,
            start_idx=breakpoint_idx,
            window_points=WINDOW_POINTS,
            min_r2=MIN_R2_LINEAR
        )

        if selected_window is not None:
            i0 = selected_window["start_idx"]
            i1 = selected_window["end_idx"]
            v0_data = df_fit.iloc[i0:i1].copy().reset_index(drop=True)
            reg = selected_window["reg"]
        else:
            v0_data = df_fit.iloc[breakpoint_idx:].copy().reset_index(drop=True)
            reg = linregress(v0_data["Time"], v0_data["Abs"])

    abs_slope = abs(reg.slope)
    v0_um_s = (abs_slope / (EPSILON * PATH_LENGTH)) * 1e6 / 60.0

    pyr_match = re.search(r'(\d+[,.]?\d*)\s*mM', filename, re.IGNORECASE)
    pyr_val = float(pyr_match.group(1).replace(',', '.')) if pyr_match else 0.0

    enzyme_match = re.search(r'(E\d+[A-Z]|Mutant[A-Z0-9]*|WT)_LDHA', filename, re.IGNORECASE)
    if not enzyme_match:
        enzyme_match = re.search(r'^(E\d+[A-Z]|Mutant[A-Z0-9]*|WT)', filename, re.IGNORECASE)
    enzyme_type = enzyme_match.group(1).upper() if enzyme_match else "Unknown"

    return {
        "filename": filename,
        "pyruvate": pyr_val,
        "v0_um_s": v0_um_s,
        "slope_abs_min": reg.slope,
        "intercept": reg.intercept,
        "r2": reg.rvalue ** 2,
        "breakpoint_time": breakpoint_time,
        "fit_start_time": float(df_fit["Time"].iloc[0]) if len(df_fit) else np.nan,
        "v0_data": v0_data,
        "full_df": df,
        "fit_df": df_fit,
        "enzyme_type": enzyme_type,
        "segmented_fit": segmented_fit,
        "selected_window": selected_window
    }

def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

st.title(f"🧪 {APP_NAME}")
st.markdown("""
Upload your kinetic CSV files. The app will automatically calculate the initial velocity ($V_0$)
from the linear portion of each curve using a segmented-regression-based method designed to skip
the initial steep slope caused by substrate addition.
""")

with st.expander(f"📜 Change Log ({APP_VERSION})", expanded=False):
    st.markdown("""
**Changes from the previous version**
- Added versioned app title.
- Added a Change Log drop-down menu.
- Improved detection of the initial steep negative slope caused by substrate addition.
- Required the pre-breakpoint slope to be steeper than the later slope.
- After finding the breakpoint, the code now tests the earliest acceptable linear window.
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
        st.subheader("📊 Results Summary & Outlier Selection")
        st.info("Uncheck the 'Include' box to exclude a run from the Michaelis-Menten fit.")

        summary_data = []
        for r in all_runs:
            summary_data.append({
                "Include": True,
                "File": r['filename'],
                "Enzyme Type": r['enzyme_type'],
                "Pyruvate (mM)": r['pyruvate'],
                "V0 (µM/s)": round(r['v0_um_s'], 4),
                "R²": round(r['r2'], 4),
                "Breakpoint (min)": round(r['breakpoint_time'], 4) if pd.notnull(r['breakpoint_time']) else np.nan
            })

        edited_df = st.data_editor(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Pyruvate (mM)": st.column_config.NumberColumn("Pyruvate (mM)", format="%.3f", disabled=False),
                "Enzyme Type": st.column_config.TextColumn("Enzyme Type", disabled=True),
                "Breakpoint (min)": st.column_config.NumberColumn("Breakpoint (min)", format="%.4f", disabled=True)
            }
        )

        included_filenames = edited_df[edited_df["Include"] == True]["File"].tolist()

        for _, row in edited_df.iterrows():
            for run in all_runs:
                if run['filename'] == row['File']:
                    run['pyruvate'] = row['Pyruvate (mM)']
                    break

        final_results = [r for r in all_runs if r['filename'] in included_filenames]

        if st.button("Calculate Michaelis-Menten Kinetics"):
            with st.expander("📝 View Detailed Calculations"):
                st.latex(r"V_0 (\mu M/s) = \frac{|\Delta Abs/min|}{\epsilon \cdot l} \cdot \frac{10^6}{60}")
                calc_table = []
                for r in final_results:
                    calc_table.append({
                        "Substrate [S]": f"{r['pyruvate']} mM",
                        "Slope (Abs/min)": round(r['slope_abs_min'], 6),
                        "Step 1: (Slope / ε)": f"{r['slope_abs_min']/EPSILON:.2e} M/min",
                        "Final V0": f"{r['v0_um_s']:.4f} µM/s",
                        "Breakpoint": f"{r['breakpoint_time']:.4f} min"
                    })
                st.table(calc_table)

            st.divider()
            st.subheader("📈 Individual Run Fits")
            cols = st.columns(2)

            for idx, r in enumerate(all_runs):
                with cols[idx % 2].expander(f"Run: {r['pyruvate']} mM Pyruvate ({r['filename']})", expanded=False):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

                    ax1.plot(r['full_df']['Time'], r['full_df']['Abs'], color='gray', alpha=0.7, label='Full trace')
                    ax1.axvline(MIN_TIME_TO_CONSIDER, color='purple', linestyle='--', label='Min time')
                    ax1.axvline(r['breakpoint_time'], color='red', linestyle='--', label='Breakpoint')
                    ax1.set_title("Full Assay")
                    ax1.set_xlabel("Time (min)")
                    ax1.set_ylabel("Abs")
                    ax1.legend()

                    fit_df = r['fit_df']
                    ax2.scatter(fit_df['Time'], fit_df['Abs'], s=10, color='lightgray', label='Eligible data')

                    if r['segmented_fit'] is not None:
                        seg = r['segmented_fit']
                        xfit = fit_df['Time'].to_numpy()
                        yfit = piecewise_linear_continuous(xfit, seg["x0"], seg["y0"], seg["m1"], seg["m2"])
                        ax2.plot(xfit, yfit, color='green', lw=1.5, label='Segmented fit')

                    ax2.scatter(r['v0_data']['Time'], r['v0_data']['Abs'], s=18, color='orange', label='Selected linear region')
                    t = r['v0_data']['Time']
                    ax2.plot(t, r['slope_abs_min'] * t + r['intercept'], color='blue', lw=1.5, label='Final linear fit')
                    ax2.axvline(r['breakpoint_time'], color='red', linestyle='--')

                    ax2.set_title(f"V0 Fit (R²={r['r2']:.4f}, [S]={r['pyruvate']} mM)")
                    ax2.set_xlabel("Time (min)")
                    ax2.set_ylabel("Abs")
                    ax2.legend()

                    st.pyplot(fig)

            if len(final_results) >= 3:
                st.divider()
                st.subheader("🧪 Michaelis-Menten Kinetic Fit")

                s_vals = np.array([r['pyruvate'] for r in final_results], dtype=float)
                v_vals = np.array([r['v0_um_s'] for r in final_results], dtype=float)

                try:
                    popt, pcov = curve_fit(
                        michaelis_menten,
                        s_vals,
                        v_vals,
                        p0=[max(v_vals), np.median(s_vals[s_vals > 0]) if np.any(s_vals > 0) else 1.0],
                        maxfev=10000
                    )
                    vmax, km = popt
                    perr = np.sqrt(np.diag(pcov))
                    vmax_err = perr[0]
                    km_err = perr[1]

                    col_res1, col_res2, col_res3 = st.columns(3)
                    col_res1.metric("Km (Michaelis Constant)", f"{km:.3f} ± {km_err:.3f} mM")
                    col_res2.metric("Vmax (Max Velocity)", f"{vmax:.3f} ± {vmax_err:.3f} µM/s")

                    plot_text = f"Km = {km:.3f} ± {km_err:.3f} mM\nVmax = {vmax:.3f} ± {vmax_err:.3f} µM/s"

                    if ENZYME_CONCENTRATION > 0:
                        kcat = vmax / ENZYME_CONCENTRATION
                        kcat_err = kcat * (vmax_err / vmax) if vmax != 0 else np.nan
                        col_res3.metric("Kcat (Turnover Number)", f"{kcat:.3f} ± {kcat_err:.3f} s⁻¹")
                        plot_text += f"\nKcat = {kcat:.3f} ± {kcat_err:.3f} s⁻¹"
                    else:
                        col_res3.metric("Kcat (Turnover Number)", "N/A (Enter enzyme conc.)")

                    fig_mm, ax_mm = plt.subplots(figsize=(6.8, 4.25))
                    s_plot = np.linspace(0, max(s_vals) * 1.2 if len(s_vals) else 1, 200)

                    ax_mm.scatter(s_vals, v_vals, color='red', label='Experimental Data', zorder=5)
                    ax_mm.plot(s_plot, michaelis_menten(s_plot, *popt), color='black', label=f'MM Fit ($K_m$={km:.2f})')
                    ax_mm.set_xlabel("Pyruvate Concentration [mM]")
                    ax_mm.set_ylabel("Initial Velocity $V_0$ [µM/s]")
                    ax_mm.legend()
                    ax_mm.grid(True, which='both', linestyle='--', alpha=0.5)

                    ax_mm.text(
                        0.95, 0.05, plot_text,
                        transform=ax_mm.transAxes,
                        fontsize=10,
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
                    )

                    st.pyplot(fig_mm)

                except Exception as e:
                    st.error(f"Could not fit Michaelis-Menten curve. Error: {e}")
            else:
                st.warning("Please include at least 3 runs to perform Michaelis-Menten fitting.")
else:
    st.write("---")
    st.info("☝️ Please upload some CSV files to begin.")

st.markdown("<p style='font-size: small;'>Roee Sela '27</p>", unsafe_allow_html=True)
