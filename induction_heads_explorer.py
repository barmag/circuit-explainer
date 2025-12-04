"""
Induction Heads & Transformer Circuits Explorer
================================================
An interactive Streamlit application for understanding Induction Heads,
a key mechanism in transformer in-context learning.

Terminology follows TransformerLens conventions.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import List, Tuple, Optional
import re

# ============================================================================
# Configuration & Page Setup
# ============================================================================

st.set_page_config(
    page_title="Induction Heads Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for AuDHD-friendly design
st.markdown("""
<style>
    /* Reduce visual clutter, increase contrast */
    .stApp {
        background-color: #0e1117;
    }

    /* Larger, clearer text */
    .big-font {
        font-size: 1.2rem !important;
        line-height: 1.8 !important;
    }

    /* Highlighted concepts */
    .concept-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #00d4ff;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* Step indicators */
    .step-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Token display */
    .token {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }

    .token-highlight {
        background: #ff6b6b;
        color: white;
    }

    .token-match {
        background: #4ecdc4;
        color: black;
    }

    .token-predict {
        background: #ffe66d;
        color: black;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }

    /* Animation keyframes hint */
    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Sidebar: Technical Depth - Q, K, V Explanations
# ============================================================================

def render_sidebar():
    """Render the technical sidebar with Q, K, V explanations."""

    with st.sidebar:
        st.markdown("## 🧠 Technical Reference")
        st.markdown("---")

        # Current interaction state
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0

        # Q, K, V Tabs
        qkv_tab = st.radio(
            "Select Component:",
            ["Query (Q)", "Key (K)", "Value (V)", "Attention Pattern"],
            horizontal=False,
            key="qkv_selector"
        )

        st.markdown("---")

        if qkv_tab == "Query (Q)":
            st.markdown("### 🔍 Query Vector")
            st.markdown("""
            <div class="concept-box">
            <b>What it represents:</b><br>
            "What am I looking for?"
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **TransformerLens notation:** `cache['q', layer]`

            **Shape:** `(batch, pos, n_heads, d_head)`

            **Intuition:** Each token generates a query vector
            that asks: *"Which previous tokens have information
            relevant to me?"*

            **In Induction Heads:**
            - The query encodes the *current token's identity*
            - It's looking for positions where this same token
              appeared before
            """)

            st.code("""
# TransformerLens example
q = cache['q', layer_idx]
# q[batch, pos, head, d_head]

# For induction head at position i:
# Q[i] ≈ "Find where token[i] appeared before"
            """, language="python")

        elif qkv_tab == "Key (K)":
            st.markdown("### 🔑 Key Vector")
            st.markdown("""
            <div class="concept-box">
            <b>What it represents:</b><br>
            "What information do I contain?"
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **TransformerLens notation:** `cache['k', layer]`

            **Shape:** `(batch, pos, n_heads, d_head)`

            **Intuition:** Each token generates a key that
            advertises: *"Here's what kind of information I have!"*

            **In Previous Token Heads:**
            - Keys are modified to encode the *previous token*
            - This is the magic! Key at position j contains
              info about token[j-1]
            """)

            st.code("""
# TransformerLens example
k = cache['k', layer_idx]
# k[batch, pos, head, d_head]

# For previous token head:
# K[j] ≈ embedding of token[j-1]
# (achieved via positional composition)
            """, language="python")

        elif qkv_tab == "Value (V)":
            st.markdown("### 💎 Value Vector")
            st.markdown("""
            <div class="concept-box">
            <b>What it represents:</b><br>
            "What should I output if selected?"
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **TransformerLens notation:** `cache['v', layer]`

            **Shape:** `(batch, pos, n_heads, d_head)`

            **Intuition:** Each token provides a value that says:
            *"If you attend to me, here's what you'll get!"*

            **In Induction Heads:**
            - Value at position j contains info to predict
              token[j+1]
            - When the induction head attends here, it copies
              the prediction for the next token
            """)

            st.code("""
# TransformerLens example
v = cache['v', layer_idx]
# v[batch, pos, head, d_head]

# For induction head attending to position j:
# V[j] → information to predict token[j+1]
            """, language="python")

        else:  # Attention Pattern
            st.markdown("### 📊 Attention Pattern")
            st.markdown("""
            <div class="concept-box">
            <b>What it represents:</b><br>
            The result of Q·K^T with softmax
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **TransformerLens notation:** `cache['pattern', layer]`

            **Shape:** `(batch, n_heads, dest_pos, src_pos)`

            **Formula:**
            """)

            st.latex(r"\text{Attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)")

            st.markdown("""
            **Interpretation:**
            - `pattern[b, h, i, j]` = how much position `i`
              attends to position `j`
            - Rows sum to 1 (softmax)
            - Causal mask: can only attend to j ≤ i
            """)

            st.code("""
# TransformerLens example
pattern = cache['pattern', layer_idx]
# pattern[batch, head, dest, src]

# Visualize:
import circuitsvis as cv
cv.attention.attention_patterns(
    tokens=tokens,
    attention=pattern[0]  # batch 0
)
            """, language="python")

        st.markdown("---")
        st.markdown("### 📚 Key Papers")
        st.markdown("""
        - [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
        - [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
        """)


# ============================================================================
# Attention Pattern Visualization
# ============================================================================

def create_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    title: str = "Attention Pattern",
    highlight_cells: Optional[List[Tuple[int, int]]] = None,
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Create an interactive Plotly heatmap for attention patterns.

    Args:
        attention_matrix: 2D numpy array of attention weights
        tokens: List of token strings
        title: Plot title
        highlight_cells: List of (row, col) tuples to highlight
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure object
    """
    n_tokens = len(tokens)

    # Create hover text with detailed information
    hover_text = []
    for i in range(n_tokens):
        row_text = []
        for j in range(n_tokens):
            text = (
                f"<b>Source:</b> '{tokens[j]}' (pos {j})<br>"
                f"<b>Dest:</b> '{tokens[i]}' (pos {i})<br>"
                f"<b>Attention:</b> {attention_matrix[i, j]:.4f}<br>"
                f"<extra></extra>"
            )
            row_text.append(text)
        hover_text.append(row_text)

    fig = go.Figure()

    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale=colorscale,
        hovertemplate="%{customdata}",
        customdata=hover_text,
        colorbar=dict(
            title=dict(
                text="Attention<br>Weight",
                side="right"
            )
        ),
        zmin=0,
        zmax=1
    ))

    # Add highlight rectangles if specified
    if highlight_cells:
        for (row, col) in highlight_cells:
            fig.add_shape(
                type="rect",
                x0=col - 0.5, x1=col + 0.5,
                y0=row - 0.5, y1=row + 0.5,
                line=dict(color="red", width=3),
                fillcolor="rgba(255, 0, 0, 0.1)"
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="white")
        ),
        xaxis=dict(
            title=dict(
                text="Source Position (attending FROM)",
                font=dict(size=14, color="white")
            ),
            tickangle=45,
            side="bottom",
            tickfont=dict(size=12, color="white")
        ),
        yaxis=dict(
            title=dict(
                text="Destination Position (attending TO)",
                font=dict(size=14, color="white")
            ),
            tickfont=dict(size=12, color="white"),
            autorange="reversed"
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=500,
        margin=dict(l=100, r=50, t=80, b=100)
    )

    return fig


def create_previous_token_pattern(tokens: List[str]) -> np.ndarray:
    """
    Create the attention pattern for a Previous Token Head.
    Each position attends to the position immediately before it.
    """
    n = len(tokens)
    pattern = np.zeros((n, n))

    for i in range(1, n):
        pattern[i, i-1] = 1.0  # Attend to previous position

    # First token has no previous, attends to itself
    pattern[0, 0] = 1.0

    return pattern


def create_induction_pattern(tokens: List[str]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Create the attention pattern for an Induction Head.
    Finds repeated tokens and attends to the position after their first occurrence.

    Returns:
        Tuple of (attention_matrix, list of key cells to highlight)
    """
    n = len(tokens)
    pattern = np.zeros((n, n))
    highlights = []

    # Normalize tokens for comparison (case-insensitive, strip punctuation)
    def normalize(t):
        return re.sub(r'[^\w]', '', t.lower())

    normalized = [normalize(t) for t in tokens]

    for i in range(n):
        attended = False
        # Look for previous occurrences of this token
        for j in range(i):
            if normalized[j] == normalized[i] and normalized[j]:
                # Found a match! Attend to position j+1 (the token AFTER the match)
                if j + 1 < i:  # Make sure j+1 is before current position
                    pattern[i, j + 1] = 1.0
                    highlights.append((i, j + 1))
                    attended = True
                    break

        if not attended:
            # Default: slight attention to self or previous
            if i > 0:
                pattern[i, i-1] = 0.3
            pattern[i, i] = 0.7 if i == 0 else 0.3

    # Normalize rows to sum to 1
    row_sums = pattern.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    pattern = pattern / row_sums

    return pattern, highlights


# ============================================================================
# Circuit Simulation Visualizations
# ============================================================================

def render_token_sequence(
    tokens: List[str],
    highlight_indices: Optional[List[int]] = None,
    match_indices: Optional[List[int]] = None,
    predict_indices: Optional[List[int]] = None
) -> str:
    """Render a sequence of tokens as styled HTML."""
    html_parts = []

    for i, token in enumerate(tokens):
        classes = ["token"]
        if highlight_indices and i in highlight_indices:
            classes.append("token-highlight")
        elif match_indices and i in match_indices:
            classes.append("token-match")
        elif predict_indices and i in predict_indices:
            classes.append("token-predict")
        else:
            classes.append("token")
            # Default background
            style = "background: #3d3d5c; color: white;"
            html_parts.append(
                f'<span class="{" ".join(classes)}" style="{style}">'
                f'[{i}] {token}</span>'
            )
            continue

        html_parts.append(f'<span class="{" ".join(classes)}">[{i}] {token}</span>')

    return " ".join(html_parts)


def create_information_flow_diagram(
    tokens: List[str],
    step: int,
    source_idx: int,
    dest_idx: int
) -> go.Figure:
    """
    Create a visual diagram showing information flow between positions.
    """
    n = len(tokens)

    fig = go.Figure()

    # Token positions (x-axis)
    x_positions = list(range(n))
    y_base = 0

    # Draw tokens as boxes
    for i, token in enumerate(tokens):
        color = "#3d3d5c"
        if i == source_idx:
            color = "#4ecdc4"  # Source: teal
        elif i == dest_idx:
            color = "#ff6b6b"  # Destination: coral

        # Token box
        fig.add_trace(go.Scatter(
            x=[i],
            y=[y_base],
            mode="markers+text",
            marker=dict(
                size=50,
                color=color,
                symbol="square",
                line=dict(color="white", width=2)
            ),
            text=[f"{token}"],
            textposition="middle center",
            textfont=dict(size=12, color="white"),
            hovertemplate=f"Position {i}<br>Token: {token}<extra></extra>",
            showlegend=False
        ))

        # Position label below
        fig.add_annotation(
            x=i, y=y_base - 0.3,
            text=f"pos {i}",
            showarrow=False,
            font=dict(size=10, color="#888")
        )

    # Draw the information flow arrow
    if step == 1:
        # Previous Token Head: info flows from source to dest (dest attends to source)
        # Arrow from source to the residual stream "above"
        fig.add_annotation(
            x=source_idx,
            y=y_base + 0.5,
            ax=dest_idx,
            ay=y_base + 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor="#00d4ff"
        )

        # Label the arrow
        mid_x = (source_idx + dest_idx) / 2
        fig.add_annotation(
            x=mid_x,
            y=y_base + 0.7,
            text="Previous Token Head<br>copies token info backward",
            showarrow=False,
            font=dict(size=11, color="#00d4ff"),
            bgcolor="rgba(0, 212, 255, 0.1)",
            borderpad=4
        )

    elif step == 2:
        # Induction Head: looking back and copying
        fig.add_annotation(
            x=source_idx,
            y=y_base + 0.5,
            ax=dest_idx,
            ay=y_base + 0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor="#ff6b6b"
        )

        mid_x = (source_idx + dest_idx) / 2
        fig.add_annotation(
            x=mid_x,
            y=y_base + 0.7,
            text="Induction Head<br>attends to token after match",
            showarrow=False,
            font=dict(size=11, color="#ff6b6b"),
            bgcolor="rgba(255, 107, 107, 0.1)",
            borderpad=4
        )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, n - 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.7, 1.2]
        ),
        height=250,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


def create_qk_composition_diagram() -> go.Figure:
    """
    Create a diagram showing Q-K composition in the induction circuit.
    """
    fig = go.Figure()

    # Layer 0: Previous Token Head
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[1, 0.5],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=15, color="#00d4ff"),
        name="Layer 0: Previous Token Head"
    ))

    fig.add_annotation(x=0, y=1, text="Token j", showarrow=False,
                       font=dict(size=12, color="white"), yshift=20)
    fig.add_annotation(x=0, y=0.5, text="Key encodes<br>token j-1", showarrow=False,
                       font=dict(size=10, color="#00d4ff"), xshift=80)

    # Layer 1: Induction Head
    fig.add_trace(go.Scatter(
        x=[1, 1],
        y=[1, 0.5],
        mode="lines+markers",
        line=dict(color="#ff6b6b", width=3),
        marker=dict(size=15, color="#ff6b6b"),
        name="Layer 1: Induction Head"
    ))

    fig.add_annotation(x=1, y=1, text="Token i", showarrow=False,
                       font=dict(size=12, color="white"), yshift=20)
    fig.add_annotation(x=1, y=0.5, text="Query looks for<br>'same as me'", showarrow=False,
                       font=dict(size=10, color="#ff6b6b"), xshift=80)

    # Connection arrow
    fig.add_annotation(
        x=0.5, y=0.5,
        ax=0.1, ay=0.5,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowcolor="#ffe66d",
        arrowwidth=2
    )

    fig.add_annotation(
        x=0.5, y=0.35,
        text="Q-K Match!<br>Query(i) · Key(j) is high<br>when token[i] == token[j-1]",
        showarrow=False,
        font=dict(size=10, color="#ffe66d"),
        bgcolor="rgba(255, 230, 109, 0.1)",
        borderpad=6
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(color="white")
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 1.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1.3]
        ),
        height=300,
        margin=dict(l=20, r=20, t=20, b=60)
    )

    return fig


# ============================================================================
# Main Application Sections
# ============================================================================

def render_introduction():
    """Render the introduction section."""

    st.markdown("## 🧩 What are Induction Heads?")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>The Core Insight</h3>
        <p class="big-font">
        Induction heads are attention heads that implement <b>in-context learning</b>
        by detecting and continuing patterns. They're one of the first discovered
        <b>circuits</b> in transformers.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### The Pattern: `[A][B] ... [A] → [B]`

        When an induction head sees a token that appeared earlier in the sequence,
        it predicts that the *next* token will be whatever came after the first occurrence.

        **Example:**
        - Sequence: `"The cat sat on the mat. The cat..."`
        - The model sees the second `"cat"` and recalls `"sat"` came after the first `"cat"`
        - Prediction: `"sat"`

        This is **not** memorization—it works for arbitrary patterns the model has
        never seen before!
        """)

    with col2:
        st.markdown("### 🎯 Quick Facts")
        st.info("""
        **Discovered:** 2022

        **Layers:** Typically L1+

        **Mechanism:** Two-head circuit

        **Function:** Pattern completion
        """)

        st.success("""
        **Why it matters:**

        Induction heads are evidence that
        transformers learn *algorithms*,
        not just statistics!
        """)


def render_circuit_simulation():
    """Render the step-by-step circuit simulation."""

    st.markdown("## 🔧 The Induction Circuit: Step by Step")

    st.markdown("""
    The induction circuit requires **two attention heads working together** across layers.
    Let's trace through how it works.
    """)

    # Example sequence
    example_tokens = ["The", "cat", "sat", ".", "The", "cat"]

    st.markdown("### 📝 Example Sequence")
    st.markdown(
        render_token_sequence(example_tokens),
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Step selector
    step = st.radio(
        "Select step to visualize:",
        [
            "Step 1: Previous Token Head (Layer 0)",
            "Step 2: Induction Head (Layer 1)",
            "Step 3: Complete Circuit"
        ],
        horizontal=True
    )

    if "Step 1" in step:
        render_step1_previous_token_head(example_tokens)
    elif "Step 2" in step:
        render_step2_induction_head(example_tokens)
    else:
        render_step3_complete_circuit(example_tokens)


def render_step1_previous_token_head(tokens: List[str]):
    """Render Step 1: Previous Token Head explanation."""

    st.markdown("""
    <div class="step-indicator">STEP 1: Previous Token Head</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### What it does:

        The **Previous Token Head** is in an early layer (typically Layer 0).
        Its job is simple but crucial:

        > **Each position's Key vector encodes information about
        > the *previous* token**

        This is achieved through the **positional embeddings**. The head learns
        to use the position offset to copy token identity backward.

        ### Why this matters:

        After this head runs, when we're at position `j`, the residual stream
        now contains information about token `j-1`. This sets up the induction head!

        ### Visual Intuition:

        Look at the attention pattern → Notice the **diagonal stripe**
        one position off! Each token attends to the one before it.
        """)

    with col2:
        # Create and display previous token attention pattern
        pattern = create_previous_token_pattern(tokens)
        fig = create_attention_heatmap(
            pattern,
            tokens,
            title="Previous Token Head Attention Pattern",
            colorscale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Information flow diagram
    st.markdown("### Information Flow:")

    # Show position 5 attending to position 4
    flow_fig = create_information_flow_diagram(tokens, step=1, source_idx=4, dest_idx=5)
    st.plotly_chart(flow_fig, use_container_width=True)

    st.markdown("""
    <div class="concept-box">
    <b>Key Insight:</b> Position 5 ("cat") now has access to information from
    position 4 ("The"). Its Key vector effectively encodes "I come after 'The'".
    </div>
    """, unsafe_allow_html=True)


def render_step2_induction_head(tokens: List[str]):
    """Render Step 2: Induction Head explanation."""

    st.markdown("""
    <div class="step-indicator">STEP 2: Induction Head</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### What it does:

        The **Induction Head** is in a later layer (typically Layer 1+).
        It performs the pattern-matching:

        > **Query asks: "Where did I appear before?"**
        > **Key answers: "The token before me was X"**

        The head looks back through the sequence for positions where
        the Key matches the Query. Thanks to the Previous Token Head,
        this finds positions where *the previous token* matches the
        *current token*.

        ### The Magic Match:

        - Current position (5): token is "cat", Query asks "find 'cat'"
        - Position 1: Key encodes "previous token was 'The'" → No match with "cat"
        - Position 2: Key encodes "previous token was 'cat'" → **MATCH!** 🎯

        ### Then What?

        The Value at position 2 contains information useful for predicting
        what comes next → "sat"!
        """)

    with col2:
        # Create and display induction attention pattern
        pattern, highlights = create_induction_pattern(tokens)
        fig = create_attention_heatmap(
            pattern,
            tokens,
            title="Induction Head Attention Pattern",
            highlight_cells=highlights,
            colorscale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Information flow diagram
    st.markdown("### Information Flow:")

    # Position 5 attends to position 2 (after the first "cat")
    flow_fig = create_information_flow_diagram(tokens, step=2, source_idx=2, dest_idx=5)
    st.plotly_chart(flow_fig, use_container_width=True)

    st.markdown("""
    <div class="concept-box">
    <b>Key Insight:</b> Position 5 ("cat" #2) attends strongly to position 2 ("sat")
    because position 2's Key (set up by the Previous Token Head) encodes "cat"
    (the token at position 1). The Value at position 2 then helps predict "sat"!
    </div>
    """, unsafe_allow_html=True)


def render_step3_complete_circuit(tokens: List[str]):
    """Render Step 3: Complete circuit overview."""

    st.markdown("""
    <div class="step-indicator">STEP 3: Complete Circuit</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### The Full Induction Circuit

    Now let's see how both heads work together!
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### Circuit Composition:

        1. **Layer 0 - Previous Token Head**
           - Writes to residual stream
           - Key[j] ← information about token[j-1]

        2. **Layer 1 - Induction Head**
           - Reads from residual stream
           - Query[i] asks "where is token[i]?"
           - Matches with Key[j] where token[j-1] == token[i]
           - Attends to position j (which is right after the match!)
           - Value[j] → prediction for next token

        ### Mathematical View:

        The **Q-K circuit** implements:
        """)

        st.latex(r"\text{Attn}[i,j] \propto \text{Query}_i \cdot \text{Key}_j")
        st.latex(r"\approx \text{embed}(\text{token}_i) \cdot \text{embed}(\text{token}_{j-1})")

        st.markdown("""
        High attention when token[i] == token[j-1]!
        """)

    with col2:
        # Show Q-K composition diagram
        fig = create_qk_composition_diagram()
        st.plotly_chart(fig, use_container_width=True)

    # Show both patterns side by side
    st.markdown("### Comparing Both Attention Patterns:")

    col1, col2 = st.columns(2)

    with col1:
        pattern1 = create_previous_token_pattern(tokens)
        fig1 = create_attention_heatmap(
            pattern1, tokens,
            "Layer 0: Previous Token Head",
            colorscale="Blues"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        pattern2, highlights = create_induction_pattern(tokens)
        fig2 = create_attention_heatmap(
            pattern2, tokens,
            "Layer 1: Induction Head",
            highlight_cells=highlights,
            colorscale="Reds"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.success("""
    **🎉 Result:** The model correctly predicts that after "The cat" (the second time),
    the next token should be "sat" — just like the first time!

    This is **in-context learning** — the model learned a pattern *within this context*
    that it may have never seen during training!
    """)


def render_playground():
    """Render the interactive playground."""

    st.markdown("## 🎮 Interactive Playground")

    st.markdown("""
    Type your own sentence to see how induction heads would activate!

    **Tip:** Include a repeated word or phrase for best results.
    Examples:
    - "Harry Potter went to school. Harry Potter"
    - "The quick brown fox. The quick"
    - "ABC ABC"
    """)

    # Text input
    user_input = st.text_input(
        "Enter your sequence:",
        value="Harry Potter went to school. Harry Potter",
        placeholder="Type a sentence with repeated tokens..."
    )

    # Tokenize (simple whitespace tokenization)
    tokens = user_input.split()

    # Error handling for short sequences
    if len(tokens) < 2:
        st.warning("⚠️ Please enter at least 2 tokens (words) to visualize attention patterns.")
        return

    if len(tokens) > 20:
        st.warning("⚠️ Sequence truncated to 20 tokens for visualization clarity.")
        tokens = tokens[:20]

    st.markdown("### 📊 Your Tokens:")
    st.markdown(render_token_sequence(tokens), unsafe_allow_html=True)

    st.markdown("---")

    # Analysis tabs
    analysis_tab = st.tabs([
        "🔍 Induction Pattern",
        "⬅️ Previous Token Pattern",
        "📈 Combined Analysis"
    ])

    with analysis_tab[0]:
        st.markdown("### Induction Head Attention Pattern")
        st.markdown("""
        **Red highlighted cells** show where the induction mechanism activates —
        where a token attends to the position *after* its previous occurrence.
        """)

        pattern, highlights = create_induction_pattern(tokens)
        fig = create_attention_heatmap(
            pattern, tokens,
            "Induction Head Pattern (Your Input)",
            highlight_cells=highlights,
            colorscale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show detected patterns
        if highlights:
            st.success(f"**✅ Detected {len(highlights)} induction pattern(s)!**")
            for (dest, src) in highlights:
                if src > 0:
                    st.markdown(
                        f"- Position **{dest}** ('{tokens[dest]}') attends to "
                        f"position **{src}** ('{tokens[src]}') — "
                        f"because '{tokens[dest]}' appeared at position **{src-1}**"
                    )
        else:
            st.info("No strong induction patterns detected. Try adding repeated words!")

    with analysis_tab[1]:
        st.markdown("### Previous Token Head Attention Pattern")
        st.markdown("""
        This shows the simple "attend to previous position" pattern that sets up
        the induction mechanism.
        """)

        pattern = create_previous_token_pattern(tokens)
        fig = create_attention_heatmap(
            pattern, tokens,
            "Previous Token Head Pattern",
            colorscale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    with analysis_tab[2]:
        st.markdown("### Combined Circuit Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Layer 0: Previous Token Head")
            pattern1 = create_previous_token_pattern(tokens)
            fig1 = create_attention_heatmap(
                pattern1, tokens,
                "Previous Token Head",
                colorscale="Blues"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("#### Layer 1: Induction Head")
            pattern2, highlights = create_induction_pattern(tokens)
            fig2 = create_attention_heatmap(
                pattern2, tokens,
                "Induction Head",
                highlight_cells=highlights,
                colorscale="Reds"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Prediction explanation
        if highlights:
            st.markdown("### 🔮 Predictions from Induction:")

            for (dest, src) in highlights:
                if src < len(tokens) - 1:
                    predicted = tokens[src] if src < len(tokens) else "???"
                    st.markdown(f"""
                    <div class="concept-box">
                    At position <b>{dest}</b>, seeing "<b>{tokens[dest]}</b>", the model would predict:
                    <br><br>
                    → If this pattern continued, next token might be similar to what came at position <b>{src}</b>: "<b>{predicted}</b>"
                    </div>
                    """, unsafe_allow_html=True)


def render_advanced_details():
    """Render advanced technical details section."""

    st.markdown("## 🔬 Advanced: OV and QK Circuits")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### QK Circuit (What to attend to)

        The QK circuit determines the attention pattern:

        ```
        Attention[i,j] = softmax(Q[i] · K[j]^T / √d)
        ```

        **For Induction Heads:**

        The QK circuit is trained such that:
        - Q encodes the current token's identity
        - K encodes the previous token's identity (via composition with L0)
        - High dot product when token[i] == token[j-1]

        **TransformerLens access:**
        ```python
        W_QK = model.W_Q[layer, head] @ model.W_K[layer, head].T
        # Shape: (d_model, d_model)
        ```
        """)

    with col2:
        st.markdown("""
        ### OV Circuit (What to output)

        The OV circuit determines what information is moved:

        ```
        Output = Attention @ V @ W_O
        ```

        **For Induction Heads:**

        The OV circuit copies the token at the attended position,
        which helps predict the next token.

        **Why position j?**

        When we attend to j (where token[j-1] == token[i]):
        - V[j] contains information about token[j]
        - This is exactly what should come next!
        - [A][B]...[A] → attend to position of [B] → output [B]

        **TransformerLens access:**
        ```python
        W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
        # Shape: (d_model, d_model)
        ```
        """)

    st.markdown("---")

    st.markdown("""
    ### 🧪 Verify Induction Heads in Real Models

    You can detect induction heads using the **prefix matching score**:

    ```python
    import transformer_lens as tl

    # Load a model
    model = tl.HookedTransformer.from_pretrained("gpt2-small")

    # Create a repeated random sequence
    # [A][B][C]...[A][B][C]...
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 50))
    repeated = torch.cat([tokens, tokens], dim=1)

    # Run with cache
    _, cache = model.run_with_cache(repeated)

    # Check attention patterns
    for layer in range(model.cfg.n_layers):
        pattern = cache['pattern', layer][0]  # (n_heads, seq, seq)

        # Induction heads should attend to position i - seq_len + 1
        # (the corresponding position in the first half)
        for head in range(model.cfg.n_heads):
            # Calculate induction score
            # ...
    ```
    """)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""

    # Render sidebar
    render_sidebar()

    # Main title
    st.markdown("""
    # 🔍 Induction Heads Explorer
    ### Understanding Transformer Circuits Through Interactive Visualization
    """)

    st.markdown("---")

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Introduction",
        "🔧 Circuit Simulation",
        "🎮 Playground",
        "🔬 Advanced Details"
    ])

    with tab1:
        render_introduction()

    with tab2:
        render_circuit_simulation()

    with tab3:
        render_playground()

    with tab4:
        render_advanced_details()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    Built with ❤️ for mechanistic interpretability learners<br>
    Terminology follows <a href="https://github.com/neelnanda-io/TransformerLens">TransformerLens</a> conventions
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
