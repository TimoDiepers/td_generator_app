import json
from uuid import uuid4

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Temporal Distribution Builder", layout="centered")

__doc__ = """
This Streamlit app replicates the functionality of the interactive Jupyter
widget provided by the user for drafting and visualizing temporal distributions.
It provides both a generator mode, where distributions can be built using
predefined shapes (uniform, triangular and normal), and a manual mode,
where users can input their own dates and amounts.  The underlying
TemporalDistribution class and helper functions are re-implemented here to
avoid external dependencies on ``bw_temporalis`` while preserving the same
API.  The generated Python code shown in the interface still refers to
``bw_temporalis`` so users can integrate the output into their own projects.
"""

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_copy_button(code_text: str) -> None:
    """Render a copy-to-clipboard button using a lightweight HTML component."""
    unique_id = uuid4().hex
    button_id = f"copy-btn-{unique_id}"
    textarea_id = f"copy-text-{unique_id}"
    escaped_code = json.dumps(code_text)
    components.html(
        f"""
        <div style="margin-top:0.5rem;">
          <button id="{button_id}"
                  style="padding:0.4rem 0.9rem; border-radius:0.5rem; border:1px solid #c3c6cf; background:#f1f3f6; cursor:pointer; font-size:0.9rem;">
            Copy code
          </button>
          <textarea id="{textarea_id}" style="position:fixed; top:-1000px; left:-1000px;"></textarea>
        </div>
        <script>
          (function() {{
            const btn = document.getElementById("{button_id}");
            const textarea = document.getElementById("{textarea_id}");
            if (!btn || !textarea) {{ return; }}
            textarea.value = {escaped_code};
            btn.addEventListener("click", function() {{
              textarea.select();
              textarea.setSelectionRange(0, textarea.value.length);
              const succeeded = document.execCommand("copy");
              const original = btn.innerText;
              btn.innerText = succeeded ? "Copied!" : "Copy failed";
              setTimeout(function() {{ btn.innerText = original; }}, 2000);
            }});
          }})();
        </script>
        """,
        height=70,
    )


def render_distribution_chart(df: pd.DataFrame, axis_label: str) -> None:
    """Render the temporal distribution as a dot chart without connecting lines."""
    chart = (
        alt.Chart(df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("axis", title=axis_label),
            y=alt.Y("amount", title="Amount"),
            tooltip=[
                alt.Tooltip("axis", title=axis_label),
                alt.Tooltip("amount", title="Amount", format=".4f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Internal TemporalDistribution implementation and helpers
# ---------------------------------------------------------------------------
RESOLUTION_LABELS = {
    "Y": "Years",
    "M": "Months",
    "D": "Days",
    "h": "Hours",
    "m": "Minutes",
    "s": "Seconds",
}


class TemporalDistribution:
    """A simplified TemporalDistribution class compatible with bw_temporalis.

    It stores date (numpy ``timedelta64`` or ``datetime64`` array) and amount
    values and exposes an ``as_dataframe`` helper to visualize the distribution.
    """

    def __init__(self, date: np.ndarray, amount: np.ndarray):
        if not isinstance(date, np.ndarray) or not isinstance(amount, np.ndarray):
            raise ValueError("`date` and `amount` must be numpy arrays")
        if date.shape != amount.shape:
            raise ValueError("`date` and `amount` must have the same shape")
        # ``timedelta64`` has dtype.kind == 'm'; ``datetime64`` has dtype.kind == 'M'
        if date.dtype.kind not in ("m", "M"):
            raise ValueError(
                f"`date` must be numpy datetime64 or timedelta64 array (got {date.dtype})"
            )
        self.date = date
        # Ensure ``amount`` is float64
        self.amount = amount.astype(np.float64)

    def as_dataframe(
        self, resolution: str | None = None
    ) -> tuple[pd.DataFrame, str]:
        """Return a DataFrame suitable for Streamlit charts and an axis label.

        Parameters
        ----------
        resolution : str, optional
            Unit to convert ``timedelta64`` values to.  Must be one of
            ``"Y", "M", "D", "h", "m", "s"`` when ``date`` has a ``timedelta64``
            dtype.  When ``date`` is ``datetime64``, this argument is ignored.
        """
        if np.issubdtype(self.date.dtype, np.datetime64):
            x_values = pd.to_datetime(self.date)
            axis_label = "Date"
        else:
            unit = resolution or "s"
            try:
                converted = self.date.astype(f"timedelta64[{unit}]")
            except Exception as exc:
                raise ValueError(
                    f"Invalid resolution for timedelta64: {resolution}"
                ) from exc
            axis_label = f"Time ({RESOLUTION_LABELS.get(unit, unit)})"
            x_values = converted.astype(np.int64)

        df = pd.DataFrame({"axis": x_values, "amount": self.amount})
        return df, axis_label


def normalized_data_array(steps: int, kind: str, param: float | None) -> np.ndarray:
    """Return an array of length ``steps`` representing a normalized distribution.

    Parameters
    ----------
    steps : int
        Number of samples in the distribution.  Must be greater than one.
    kind : str
        One of ``"uniform"``, ``"triangular"``, or ``"normal"``.
    param : float | None
        Parameter controlling the distribution shape.  For ``"uniform"``
        distributions this is ignored.  For ``"triangular"`` it represents the
        mode in the range [0, 1], and defaults to 0.5 when ``None``.  For
        ``"normal"`` it represents the standard deviation of a normal
        distribution centered at 0 on the interval [-0.5, 0.5].

    Returns
    -------
    numpy.ndarray
        An array of shape ``(steps,)`` whose entries sum to one.
    """
    if kind == "uniform":
        values = np.ones(steps)
    elif kind == "triangular":
        # Mode between 0 and 1.  Default to the center for an isosceles triangle.
        c = 0.5 if param is None else float(param)
        if not 0 <= c <= 1:
            raise ValueError(f"`param` must be within [0, 1] for triangular distributions (got {c})")
        x = np.linspace(0, 1, steps)
        # Piecewise linear triangular PDF on [0, 1]
        values = np.where(
            x < c,
            # Avoid division by zero if c == 0
            np.where(c == 0, 0, 2 * x / c),
            # Avoid division by zero if c == 1
            np.where(c == 1, 0, 2 * (1 - x) / (1 - c)),
        )
    elif kind == "normal":
        # Standard deviation must be strictly positive
        if param is None or param <= 0:
            raise ValueError(
                "Parameter must be a positive number for normal distributions"
            )
        x = np.linspace(-0.5, 0.5, steps)
        # Gaussian PDF with mean 0 and standard deviation ``param``
        values = np.exp(-0.5 * (x / param) ** 2)
    else:
        raise ValueError(f"Unrecognized distribution kind: {kind}")

    # Remove any non-real values (should not occur) and normalize
    mask = np.isreal(values)
    values = values[mask]
    total = values.sum()
    if total == 0:
        # Prevent division by zero if the PDF collapsed to zeros
        return np.ones_like(values) / len(values)
    return values / total


def easy_timedelta_distribution(
    start: int,
    end: int,
    resolution: str,
    steps: int,
    kind: str = "uniform",
    param: float | None = None,
) -> TemporalDistribution:
    """Generate a timedelta ``TemporalDistribution`` from simple inputs.

    Parameters
    ----------
    start : int
        Start of the range, inclusive, in units of ``resolution``.
    end : int
        End of the range, inclusive, in units of ``resolution``.
    resolution : str
        One of ``"Y", "M", "D", "h", "m", "s"`` indicating the time unit.
    steps : int
        Number of discrete points in the distribution (must be greater than one).
    kind : str
        Type of distribution.  Must be one of ``"uniform"``, ``"triangular"``,
        or ``"normal"``.  Defaults to ``"uniform"``.
    param : float | None
        Optional parameter controlling the distribution shape.  Ignored for
        uniform distributions.  For triangular distributions, ``param`` is the
        mode in the same units as ``start`` and ``end``.  For normal
        distributions it is the standard deviation on the standardized domain.

    Returns
    -------
    TemporalDistribution
        A TemporalDistribution instance containing the generated date and amount
        arrays.
    """
    # Validate inputs
    if not isinstance(steps, int) or steps <= 1:
        raise ValueError(f"`steps` must be an integer greater than one; got {steps}")
    if resolution not in "YMDhms":
        raise ValueError(f"Invalid resolution '{resolution}'; must be one of Y, M, D, h, m, s")
    if start >= end:
        raise ValueError(f"Start value is not less than end value: {start} >= {end}")
    if kind == "triangular" and steps < 3:
        raise ValueError("Triangular distribution must have at least three steps")

    # Normalize triangular parameter to [0, 1] range when provided in absolute units
    param_norm: float | None = None
    if kind == "triangular" and param is not None:
        # Convert to fraction of the interval [start, end]
        if end - start == 0:
            raise ValueError("Start and end cannot be equal for triangular distributions")
        param_norm = (param - start) / (end - start)
        if not 0 <= param_norm <= 1:
            raise ValueError("Triangular mode is outside (start, end) bounds")

    # Generate date array evenly spaced between start and end, inclusive
    date = np.linspace(start, end, steps).astype(f"timedelta64[{resolution}]")
    # Compute distribution values
    values = normalized_data_array(steps, kind, param_norm if kind == "triangular" else param)
    # Filter out non-real values (unlikely) and normalize
    mask = np.isreal(values)
    values = values[mask]
    date = date[mask]
    if values.sum() != 0:
        values = values / values.sum()
    return TemporalDistribution(date=date, amount=values)


# ---------------------------------------------------------------------------
# Utility functions for parsing user input and generating code snippets
# ---------------------------------------------------------------------------
def parse_num_list(text: str, label: str) -> list[float]:
    """Parse a comma- or whitespace-separated list of numbers from ``text``.

    Both integers and floating point numbers are accepted.  Raises a
    ``ValueError`` if parsing fails or the list is empty.
    """
    parts = [p for p in text.replace(",", " ").split() if p]
    if not parts:
        raise ValueError(f"{label} cannot be empty.")
    values: list[float] = []
    for p in parts:
        try:
            # Interpret values containing a decimal point or scientific notation as floats
            if ("." in p) or ("e" in p.lower()):
                values.append(float(p))
            else:
                values.append(int(p))
        except Exception as exc:
            raise ValueError(f"Could not parse '{p}' in {label}.") from exc
    return values


def format_number(value: float) -> str:
    """Format ``value`` as a compact string.

    Whole numbers are rendered without a decimal point; other numbers have
    trailing zeros stripped.
    """
    as_float = float(value)
    if np.isfinite(as_float) and as_float.is_integer():
        return str(int(as_float))
    return (f"{as_float:.6f}").rstrip("0").rstrip(".")


def build_code_generator(
    start: int,
    end: int,
    resolution: str,
    steps: int,
    kind: str,
    param: float | None,
) -> str:
    """Return a string containing Python code to construct a generator distribution."""
    s, e = sorted([start, end])
    k = kind
    p = param
    # Build the call to ``easy_timedelta_distribution``
    lines = [
        "td = easy_timedelta_distribution(",
        f"    start={s},",
        f"    end={e},",
        f"    resolution='{resolution}',",
        f"    steps={steps},",
        f"    kind='{k}'",
    ]
    if p is not None:
        lines[-1] += ","  # Trailing comma on the previous line
        lines.append(f"    param={format_number(p)}")
    lines.append(")")
    return "\n".join(lines)


def build_code_manual(
    dates: list[float], amounts: list[float], unit: str
) -> str:
    """Return a string containing Python code to construct a manual distribution."""
    d_str = ", ".join(format_number(x) for x in dates)
    a_str = ", ".join(format_number(x) for x in amounts)
    return (
        f"date = np.array([{d_str}], dtype='timedelta64[{unit}]')\n"
        f"amount = np.array([{a_str}], dtype=float)\n"
        "td = TemporalDistribution(date=date, amount=amount)"
    )


def build_code(
    mode: str,
    start: int,
    end: int,
    resolution: str,
    steps: int,
    kind: str,
    param: float | None,
    manual_unit: str,
    dates: list[float],
    amounts: list[float],
    include_imports: bool = False,
) -> str:
    """Return a complete code snippet for the current mode.

    If ``include_imports`` is True, the necessary import statements for
    bw_temporalis are included at the top of the snippet.
    """
    if mode == "Generator":
        body = build_code_generator(
            start=start,
            end=end,
            resolution=resolution,
            steps=steps,
            kind=kind,
            param=param,
        )
        imports = ["from bw_temporalis import easy_timedelta_distribution"]
    else:
        body = build_code_manual(dates=dates, amounts=amounts, unit=manual_unit)
        imports = ["import numpy as np", "from bw_temporalis import TemporalDistribution"]

    if not include_imports:
        return body
    return "\n".join(imports + ["", body])


# ---------------------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------------
st.title("Temporal Distribution Builder")

st.write(
    "Use this app to draft and visualize the temporal distributions required by [bw_temporalis](https://github.com/brightway-lca/bw_temporalis) and [bw_timex](https://github.com/brightway-lca/bw_timex)."
)

tab_generator, tab_manual = st.tabs(["Generator", "Manual"])

with tab_generator:
    kind_val = st.radio("distribution kind", ["uniform", "triangular", "normal"], horizontal=True)

    # Generator controls
    c1, c2, c3 = st.columns((1, 1, 1))
    with c1:
        start_val = st.number_input("start", value=0, step=1)
    with c2:
        end_val = st.number_input("end", value=10, step=1)
    with c3:
        # Human-readable labels mapping to resolution codes
        gen_resolutions = {
            "Years": "Y",
            "Months": "M",
            "Days": "D",
        }
        resolution_label = st.selectbox(
            "resolution",
            list(gen_resolutions.keys()),
            index=0,
        )
        resolution_code = gen_resolutions[resolution_label]

    max_steps = max(1, int(end_val - start_val) + 1)
    min_steps = 1 if max_steps < 2 else 2
    steps_key = "generator_steps_value"
    if steps_key not in st.session_state:
        st.session_state[steps_key] = 11
    st.session_state[steps_key] = max(
        min_steps, min(st.session_state[steps_key], max_steps)
    )
    stored_steps = st.session_state[steps_key]

    steps_col, param_col = st.columns((1, 1))
    with steps_col:
        steps_val = st.slider(
            "steps",
            min_value=min_steps,
            max_value=max_steps,
            value=stored_steps,
            step=1,
            key=steps_key,
        )
        steps_val = st.session_state[steps_key]

    param_val: float | None = None
    with param_col:
        if kind_val == "uniform":
            pass
        elif kind_val == "triangular":
            s_int, e_int = int(start_val), int(end_val)
            if s_int == e_int:
                param_col.info(
                    "Start and end are equal; the triangular distribution collapses "
                    "to a single point and no mode can be chosen."
                )
                param_val = float(s_int)
            else:
                pmin = float(min(s_int, e_int))
                pmax = float(max(s_int, e_int))
                # Choose the midpoint as a reasonable default
                pdefault = (pmin + pmax) / 2.0
                step_size = max((pmax - pmin) / 20.0, 0.01)
                param_val = param_col.slider(
                    "mode",
                    min_value=pmin,
                    max_value=pmax,
                    value=pdefault,
                    step=step_size,
                )
        else:  # normal distribution
            param_val = param_col.slider(
                "std dev",
                min_value=0.02,
                max_value=1.0,
                value=0.15,
                step=0.01,
            )

    # Compute and display the distribution
    try:
        td_instance = easy_timedelta_distribution(
            start=int(start_val),
            end=int(end_val),
            resolution=resolution_code,
            steps=int(steps_val),
            kind=kind_val,
            param=param_val,
        )
        df, axis_label = td_instance.as_dataframe(resolution=resolution_code)
    except Exception as exc:
        st.error(f"Error: {exc}")
        td_instance = None
        df = None
        axis_label = ""

    if td_instance is not None and df is not None:
        chart_col, code_col = st.columns((3, 2))
        with chart_col:
            render_distribution_chart(df, axis_label)
        with code_col:
            include_key = "include_imports_generator"
            code_placeholder = st.empty()
            controls_cols = st.columns((1, 1))
            with controls_cols[1]:
                include_imports_value = st.checkbox(
                    "Include imports",
                    key=include_key,
                )
            code_text = build_code(
                mode="Generator",
                start=int(start_val),
                end=int(end_val),
                resolution=resolution_code,
                steps=int(steps_val),
                kind=kind_val,
                param=param_val,
                manual_unit="",
                dates=[],
                amounts=[],
                include_imports=include_imports_value,
            )
            code_placeholder.code(code_text, language="python")
            with controls_cols[0]:
                render_copy_button(code_text)

with tab_manual:
    man_resolutions = {
        "Years": "Y",
        "Months": "M",
        "Days": "D",
    }
    manual_label = st.selectbox(
        "resolution",
        list(man_resolutions.keys()),
        index=0,
        key="manual_resolution",
    )
    manual_unit_code = man_resolutions[manual_label]

    dates_text = st.text_area(
        "dates (comma or space separated)",
        value="0, 2, 4, 6, 8, 10",
        height=70,
        key="manual_dates",
    )
    amounts_text = st.text_area(
        "amounts (comma or space separated)",
        value="0.1, 0.1, 0.2, 0.2, 0.2, 0.2",
        height=70,
        key="manual_amounts",
    )

    try:
        date_vals = parse_num_list(dates_text, "dates")
        amount_vals = parse_num_list(amounts_text, "amounts")
        if len(date_vals) != len(amount_vals):
            raise ValueError("The number of dates and amounts must match.")
        if not date_vals:
            raise ValueError("Provide at least one date and amount.")
        date_array = np.array(date_vals, dtype=f"timedelta64[{manual_unit_code}]")
        amount_array = np.array(amount_vals, dtype=float)
        if np.any(np.isnan(amount_array)):
            raise ValueError("Amounts must be numeric values.")
        td_instance = TemporalDistribution(date=date_array, amount=amount_array)
        df, axis_label = td_instance.as_dataframe(resolution=manual_unit_code)
    except Exception as exc:
        st.error(f"Error: {exc}")
        td_instance = None
        df = None
        axis_label = ""

    if td_instance is not None and df is not None:
        chart_col, code_col = st.columns((3, 2))
        with chart_col:
            render_distribution_chart(df, axis_label)
        with code_col:
            include_key = "include_imports_manual"
            code_placeholder = st.empty()
            controls_cols = st.columns((1, 1))
            with controls_cols[1]:
                include_imports_value = st.checkbox(
                    "Include imports",
                    key=include_key,
                )
            code_text = build_code(
                mode="Manual",
                start=0,
                end=0,
                resolution="",
                steps=0,
                kind="",
                param=None,
                manual_unit=manual_unit_code,
                dates=date_vals,
                amounts=amount_vals,
                include_imports=include_imports_value,
            )
            code_placeholder.code(code_text, language="python")
            with controls_cols[0]:
                render_copy_button(code_text)
