# app_relu.py
# Streamlit app: Visualise the Rectified Linear Unit (ReLU) activation function

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function", layout="wide")

st.title("1. Rectified Linear Unit (ReLU)")
st.write(
    """
    The **ReLU** activation function is defined as:

    $$f(x) = \\max(0, x)$$

    It outputs the input directly if it is positive, otherwise it outputs zero.
    ReLU is the most widely used activation in deep neural networks because it is
    simple, cheap to compute, and helps mitigate the vanishing gradient problem.
    """
)

# --- Sidebar controls ---
st.sidebar.header("Settings")
x_min = st.sidebar.slider("x-axis minimum", -20.0, 0.0, -6.0, 0.5)
x_max = st.sidebar.slider("x-axis maximum", 0.0, 20.0, 6.0, 0.5)
n_points = st.sidebar.slider("Number of points", 100, 2000, 400, 50)
show_derivative = st.sidebar.checkbox("Show derivative (step function)", value=False)

# --- Compute values ---
x = np.linspace(x_min, x_max, n_points)
y = np.maximum(0, x)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="crimson", linewidth=2.5, label="ReLU(x) = max(0, x)")

if show_derivative:
    # derivative of ReLU: 1 for x > 0, 0 for x < 0 (undefined at 0)
    dy = (x > 0).astype(float)
    ax.plot(x, dy, color="navy", linewidth=2, linestyle="--",
            label="d/dx ReLU(x)")

ax.axhline(0, color="black", linewidth=0.6)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("ReLU Activation Function")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

st.pyplot(fig)

# --- Property table ---
st.subheader("Properties")
st.table({
    "Property": ["Range", "Derivative", "Zero-centred?", "Vanishing gradient?"],
    "Value": [
        "[0, +∞)",
        "0 if x<0, 1 if x>0 (undefined at 0)",
        "No",
        "No (for x>0)",
    ],
})

# --- Interactive single-value calculator ---
st.subheader("Try a value")
xv = st.number_input("Enter x", value=2.0, step=0.1)
st.write(f"**ReLU({xv}) = {max(0.0, xv):.4f}**")