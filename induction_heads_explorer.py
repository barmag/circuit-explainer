"""
Induction Heads & Transformer Circuits Explorer
================================================
An interactive Streamlit application for understanding Induction Heads,
a key mechanism in transformer in-context learning.

Terminology follows TransformerLens conventions.

Version 2.0 - Enhanced with:
- Corrected axis labels and terminology
- Circuit generalisation framework
- Composition types explanation
- Eigenvalue-based head detection
- Path expansion visualisation
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

    /* Warning/important boxes */
    .warning-box {
        background: linear-gradient(135deg, #2e1a1a 0%, #3e1616 100%);
        border-left: 4px solid #ff6b6b;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* Success/insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #1a2e1a 0%, #163e16 100%);
        border-left: 4px solid #4ecdc4;
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

    /* Code blocks */
    .code-highlight {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 0.5rem;
        font-family: 'Courier New', monospace;
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
            ["Query (Q)", "Key (K)", "Value (V)", "Attention Pattern", "Composition"],
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

            **In the Induction Circuit (K-Composition):**
            
            ⚠️ The key insight: In Layer 1, the Key is computed
            from the **residual stream**, which has been modified
            by the Layer 0 previous-token head!
            
            So K[j] in Layer 1 contains info about token[j-1]
            *because* the previous-token head wrote that info
            into position j's residual stream.
            """)

            st.code("""
# The K-composition mechanism:

# Layer 0: Previous token head
# - Attends from position j to position j-1  
# - Writes token[j-1] info into residual[j]

# Layer 1: Induction head reads residual
k = W_K @ residual[j]
# Now k encodes token[j-1] info!

# This is K-COMPOSITION:
# Head1's Keys depend on Head0's output
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
            - Value at position j contains info about token[j]
            - The OV circuit is a "copying" matrix (positive eigenvalues)
            - When attended to, it increases the logit of token[j]
            """)

            st.code("""
# TransformerLens example
v = cache['v', layer_idx]
# v[batch, pos, head, d_head]

# The OV circuit determines output:
W_OV = W_V @ W_O  # Combined matrix

# For copying heads:
# W_OV has positive eigenvalues
# → attending to token X increases P(X)
            """, language="python")

        elif qkv_tab == "Attention Pattern":
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

            **Axis Convention (IMPORTANT!):**
            - **Rows (dest_pos)**: The position doing the attending
            - **Columns (src_pos)**: The position being attended to
            
            So `pattern[b, h, i, j]` = how much position `i` 
            attends to position `j`

            **Formula:**
            """)

            st.latex(r"\text{Attn}_{i,j} = \text{softmax}\left(\frac{Q_i \cdot K_j^T}{\sqrt{d_k}}\right)")

            st.markdown("""
            **Interpretation:**
            - Rows sum to 1 (softmax over source positions)
            - Causal mask: can only attend to j ≤ i
            """)

            st.code("""
# TransformerLens example
pattern = cache['pattern', layer_idx]
# pattern[batch, head, dest, src]

# Row i shows: where does position i look?
# Column j shows: who looks at position j?

# Visualize with circuitsvis:
import circuitsvis as cv
cv.attention.attention_patterns(
    tokens=tokens,
    attention=pattern[0]  # batch 0
)
            """, language="python")

        else:  # Composition
            st.markdown("### 🔗 Attention Head Composition")
            st.markdown("""
            <div class="concept-box">
            <b>The key insight:</b><br>
            Heads in later layers read from the residual stream,
            which earlier heads have modified!
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **Three Types of Composition:**

            | Type | What's Affected | Mechanism |
            |------|----------------|-----------|
            | **Q-Comp** | Where to attend | Head0 → Head1's Query |
            | **K-Comp** | Where to attend | Head0 → Head1's Key |
            | **V-Comp** | What to output | Head0 → Head1's Value |

            **Induction heads use K-Composition:**
            
            1. Previous-token head (L0) writes token[j-1] info 
               into residual stream at position j
            2. Induction head (L1) computes Keys from this 
               modified residual stream
            3. Result: K[j] encodes token[j-1]!
            """)

            st.code("""
# Measuring composition strength:
# (from the Anthropic paper)

# K-composition between head0 and head1:
W_K_eff = W_K[layer1, head1] @ W_O[layer0, head0]

# Frobenius norm ratio indicates composition:
composition_score = (
    np.linalg.norm(W_K_eff, 'fro') / 
    (np.linalg.norm(W_K[layer1, head1], 'fro') * 
     np.linalg.norm(W_O[layer0, head0], 'fro'))
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
        attention_matrix: 2D numpy array of attention weights [dest, src]
        tokens: List of token strings
        title: Plot title
        highlight_cells: List of (row, col) tuples to highlight
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure object
    
    Note on axes:
        - Rows (y-axis) = Destination position (the one doing the attending)
        - Columns (x-axis) = Source position (being attended to)
        - attention_matrix[i, j] = how much position i attends to position j
    """
    n_tokens = len(tokens)

    # Create hover text with detailed information
    hover_text = []
    for i in range(n_tokens):
        row_text = []
        for j in range(n_tokens):
            text = (
                f"<b>Dest (attending):</b> '{tokens[i]}' (pos {i})<br>"
                f"<b>Source (attended to):</b> '{tokens[j]}' (pos {j})<br>"
                f"<b>Attention weight:</b> {attention_matrix[i, j]:.4f}<br>"
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
                text="Source Position (being attended TO)",
                font=dict(size=14, color="white")
            ),
            tickangle=45,
            side="bottom",
            tickfont=dict(size=12, color="white")
        ),
        yaxis=dict(
            title=dict(
                text="Destination Position (doing the attending)",
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
    
    The induction pattern: [A][B] ... [A] → attends to [B]
    
    When we see token[i] = A, we look for previous positions j where:
    - token[j-1] = A (the token BEFORE position j matches current token)
    - Then we attend to position j (which is [B], the token AFTER the match)
    
    This is what K-composition enables: the Key at position j encodes
    information about token[j-1], so we can match on the previous token.

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
        # Look for positions j where token[j-1] matches token[i]
        # We want to attend to j (the position AFTER the match)
        for j in range(1, i):  # j must have a previous token and be before i
            # K-composition magic: we match current token with PREVIOUS token at j
            if normalized[j-1] == normalized[i] and normalized[i]:
                # Found a match! token[j-1] == token[i]
                # Attend to position j (the token that came AFTER the previous A)
                pattern[i, j] = 1.0
                highlights.append((i, j))
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
            text="Previous Token Head<br>writes prev token info<br>into residual stream",
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
            text="Induction Head<br>attends via K-composition<br>(Keys encode prev tokens)",
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
    fig.add_annotation(x=0, y=0.5, text="Writes token[j-1]<br>into residual[j]", showarrow=False,
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

    # Connection arrow showing composition
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
        x=0.5, y=0.3,
        text="K-Composition!<br>Key[j] = W_K @ residual[j]<br>residual[j] contains token[j-1] info<br>∴ Key[j] encodes token[j-1]",
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
        height=350,
        margin=dict(l=20, r=20, t=20, b=60)
    )

    return fig


def create_path_expansion_diagram() -> go.Figure:
    """
    Create a diagram showing the path expansion of a 2-layer transformer.
    """
    fig = go.Figure()
    
    # Positions for nodes
    layers = {
        'input': 0,
        'L0': 1,
        'L1': 2,
        'output': 3
    }
    
    # Draw nodes
    nodes = [
        ('input', 0.5, 'W_E', '#888'),
        ('L0', 0.3, 'Head 0a', '#00d4ff'),
        ('L0', 0.7, 'Head 0b', '#00d4ff'),
        ('L1', 0.3, 'Head 1a', '#ff6b6b'),
        ('L1', 0.7, 'Head 1b', '#ff6b6b'),
        ('output', 0.5, 'W_U', '#888'),
    ]
    
    for layer, y, label, color in nodes:
        x = layers[layer]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=40, color=color, line=dict(color='white', width=2)),
            text=[label],
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw paths (simplified - just showing key paths)
    paths = [
        # Direct path
        ([(0, 0.5), (3, 0.5)], '#666', 'Direct (bigram)', 1),
        # Single head paths
        ([(0, 0.5), (1, 0.3), (3, 0.5)], '#00d4ff', 'Head 0a alone', 2),
        ([(0, 0.5), (2, 0.7), (3, 0.5)], '#ff6b6b', 'Head 1b alone', 2),
        # Composition path (induction!)
        ([(0, 0.5), (1, 0.3), (2, 0.7), (3, 0.5)], '#ffe66d', 'Composed (induction!)', 3),
    ]
    
    for points, color, name, width in paths:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=color, width=width),
            name=name,
            hoverinfo='name'
        ))
    
    fig.update_layout(
        title=dict(text="Path Expansion: Transformer as Sum of Paths", 
                   font=dict(color='white', size=14)),
        showlegend=True,
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.5, 3.5]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[0, 1]
        ),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add layer labels
    for layer, x in layers.items():
        label = {'input': 'Embed', 'L0': 'Layer 0', 'L1': 'Layer 1', 'output': 'Unembed'}[layer]
        fig.add_annotation(
            x=x, y=-0.1, text=label,
            showarrow=False, font=dict(color='#888', size=11)
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

        **Layers:** Requires 2+ layers

        **Mechanism:** K-Composition

        **Function:** Pattern completion
        """)

        st.success("""
        **Why it matters:**

        Induction heads are evidence that
        transformers learn *algorithms*,
        not just statistics!
        """)


def render_circuit_concept():
    """Render the 'What is a Circuit?' section - the key to generalisation."""
    
    st.markdown("## 🧠 What Is a Circuit?")
    
    st.markdown("""
    <div class="concept-box">
    <h3>The Core Framework</h3>
    <p class="big-font">
    A <b>circuit</b> is a subset of model components that work together 
    to implement a recognisable computation. The induction head is just 
    <i>one example</i> of a circuit.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### The Path Expansion View")
    
    st.markdown("""
    The key insight from the Anthropic paper: a transformer's output is a 
    **sum of paths**, where each path is an independent computational unit.
    """)
    
    st.latex(r"""
    \text{logits} = W_U \cdot \left( 
        \underbrace{W_E \cdot x}_{\text{direct path}} 
        + \sum_{h \in L_0} \underbrace{\text{head}_h(x)}_{\text{Layer 0 heads}}
        + \sum_{h \in L_1} \underbrace{\text{head}_h(x)}_{\text{Layer 1 heads}}
        + \underbrace{\text{composition terms}}_{\text{virtual heads}}
    \right)
    """)
    
    st.markdown("""
    Each path can be understood independently. A **circuit** emerges when 
    multiple paths conspire to implement something meaningful.
    """)
    
    # Path expansion diagram
    fig = create_path_expansion_diagram()
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### The Two Separable Circuits in Every Attention Head")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h4>🎯 QK Circuit</h4>
        <p><b>Question:</b> "Where should I attend?"</p>
        <p><b>Form:</b> Bilinear form on tokens</p>
        <p><b>Matrix:</b> W_Q @ W_K.T</p>
        <p><b>Output:</b> Attention pattern</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="concept-box">
        <h4>📤 OV Circuit</h4>
        <p><b>Question:</b> "What should I output?"</p>
        <p><b>Form:</b> Linear map on tokens</p>
        <p><b>Matrix:</b> W_V @ W_O</p>
        <p><b>Output:</b> Contribution to residual</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Key Insight:</b> These are completely separable! You can understand 
    <i>where</i> a head attends independently from <i>what</i> it outputs 
    when attending there.
    </div>
    """, unsafe_allow_html=True)


def render_composition_types():
    """Render the three types of attention head composition."""
    
    st.markdown("## 🔗 The Three Types of Composition")
    
    st.markdown("""
    When attention heads in different layers interact, there are exactly 
    **three ways** they can compose. Understanding these is key to finding 
    circuits in any transformer.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>Q-Composition</h3>
        <p><b>Head₀'s output → Head₁'s Query</b></p>
        <hr>
        <p><b>Effect:</b> Changes WHERE head₁ attends, 
        based on information head₀ moved.</p>
        <p><b>Example:</b> "Attend to the subject of 
        the previous clause" (requires knowing 
        where the clause boundary is).</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h3>K-Composition ⭐</h3>
        <p><b>Head₀'s output → Head₁'s Key</b></p>
        <hr>
        <p><b>Effect:</b> Changes what positions 
        "advertise" to head₁'s queries.</p>
        <p><b>Example:</b> <i>Induction heads!</i> 
        Keys at position j encode token[j-1] 
        because the previous-token head wrote 
        that info into the residual stream.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="concept-box">
        <h3>V-Composition</h3>
        <p><b>Head₀'s output → Head₁'s Value</b></p>
        <hr>
        <p><b>Effect:</b> Creates "virtual attention 
        heads" — composing two movements.</p>
        <p><b>Example:</b> Copy the token from 
        two positions back (compose two 
        previous-token heads).</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Why K-Composition is Special for Induction")
    
    st.markdown("""
    The induction circuit specifically exploits **K-composition**. Here's why:
    
    1. **Problem:** We want to find positions where the *previous* token matches our current token
    2. **Solution:** Have an earlier head write "previous token info" into each position's residual stream
    3. **Result:** When the induction head computes Keys, they encode the *previous* token!
    
    This is the "clever trick" that makes induction possible with just 2 layers.
    """)
    
    # Q-K composition diagram
    fig = create_qk_composition_diagram()
    st.plotly_chart(fig, use_container_width=True)


def render_eigenvalue_analysis():
    """Render the eigenvalue-based head detection section."""
    
    st.markdown("## 🔬 Detecting Head Types via Matrix Analysis")
    
    st.markdown("""
    How do we know if a head is "copying" or "suppressing"? The eigenvalues 
    of its OV matrix reveal its computational role!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### OV Matrix Eigenvalues
        
        The OV matrix (W_V @ W_O) determines what happens when a head 
        attends to a token:
        
        | Eigenvalue Pattern | Behavior |
        |-------------------|----------|
        | **Mostly positive** | Copying: increases logit of attended token |
        | **Mostly negative** | Suppression: decreases logit |
        | **Mixed** | Complex transformation |
        
        **For induction heads:**
        - OV circuit has positive eigenvalues (copying)
        - When attending to token X, increase P(X)
        """)
    
    with col2:
        st.markdown("""
        ### QK Matrix Analysis
        
        The QK matrix (W_Q @ W_K.T) determines attention patterns:
        
        - **Positive eigenvalues** → tends to attend to "similar" tokens
        - **Diagonal dominance** → attends to same token type
        
        **For the induction QK circuit (with K-composition):**
        
        The effective QK product matches current token 
        with *previous* token at each position, enabling 
        the [A][B]...[A] → [B] pattern.
        """)
    
    st.markdown("### Visualising Eigenvalue Distributions")
    
    # Create example eigenvalue distributions
    np.random.seed(42)
    
    # Copying head: mostly positive
    copying_eigs = np.abs(np.random.randn(64)) * 0.5 + 0.3
    
    # Suppression head: mostly negative  
    suppression_eigs = -np.abs(np.random.randn(64)) * 0.5 - 0.1
    
    # Random head: mixed
    random_eigs = np.random.randn(64) * 0.5
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "Copying Head (Induction)", "Suppression Head", "Random Head"
    ])
    
    fig.add_trace(
        go.Histogram(x=copying_eigs, nbinsx=20, marker_color='#4ecdc4', name='Copying'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=suppression_eigs, nbinsx=20, marker_color='#ff6b6b', name='Suppression'),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=random_eigs, nbinsx=20, marker_color='#888', name='Random'),
        row=1, col=3
    )
    
    # Add zero line
    for col in [1, 2, 3]:
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=col)
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.code("""
# TransformerLens: Compute OV eigenvalues
W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
eigenvalues = np.linalg.eigvals(W_OV.detach().numpy())

# Copying score: fraction of positive eigenvalues
copying_score = (eigenvalues.real > 0).mean()
print(f"Copying score: {copying_score:.2f}")  # Induction heads: > 0.8
    """, language="python")


def render_circuit_simulation():
    """Render the step-by-step circuit simulation."""

    st.markdown("## 🔧 The Induction Circuit: Step by Step")

    st.markdown("""
    The induction circuit requires **two attention heads working together** across layers,
    connected via **K-composition**. Let's trace through the mechanism.
    """)

    # Example sequence
    example_tokens = ["The", "cat", "sat", ".", "The", "cat"]

    st.markdown("### 📝 Example Sequence")
    st.markdown(
        render_token_sequence(example_tokens),
        unsafe_allow_html=True
    )
    
    st.markdown("""
    **Goal:** When we reach position 5 ("cat"), predict "sat" 
    because we saw "cat sat" earlier.
    """)

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
    <div class="step-indicator">STEP 1: Previous Token Head (Layer 0)</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### What it does:

        The **Previous Token Head** is in Layer 0. Its job is simple but crucial:

        > **It attends to the previous position and copies that token's 
        > information into the current position's residual stream.**

        ### The Mechanism:

        1. **Attention pattern:** Position j attends to position j-1
        2. **OV circuit:** Copies token embedding information
        3. **Result:** residual[j] now contains info about token[j-1]

        ### Why this matters (K-Composition setup):

        After this head runs, when Layer 1 computes Keys:
        - K[j] = W_K @ residual[j]
        - residual[j] contains token[j-1] info
        - **Therefore K[j] encodes token[j-1]!**

        This is the magic that enables induction.
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
    <b>Key Insight:</b> After this head runs, position 5's residual stream 
    contains information about "The" (the token at position 4). 
    When Layer 1 computes Keys, K[5] will encode "The"!
    </div>
    """, unsafe_allow_html=True)


def render_step2_induction_head(tokens: List[str]):
    """Render Step 2: Induction Head explanation."""

    st.markdown("""
    <div class="step-indicator">STEP 2: Induction Head (Layer 1)</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### What it does:

        The **Induction Head** is in Layer 1. It performs pattern-matching 
        using the K-composed keys:

        > **Query asks: "Where is my current token?"**
        > **Key (K-composed) answers: "The token before me was X"**

        ### The K-Composition Magic:

        Thanks to the Previous Token Head:
        - K[j] encodes token[j-1] (not token[j]!)
        - Q[i] encodes token[i]
        - High attention when token[i] == token[j-1]
        - This means: attend to position *after* where current token appeared!

        ### Concrete Example:

        At position 5 (second "cat"):
        - Q[5] encodes "cat"
        - K[2] encodes "cat" (because token[1] = "cat")
        - **Match!** Position 5 attends to position 2
        - V[2] contains info about "sat" → predict "sat"!
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
    <b>Key Insight:</b> Position 5 ("cat" #2) attends to position 2 ("sat")
    because K[2] encodes "cat" (the previous token). The OV circuit then 
    copies information useful for predicting "sat"!
    </div>
    """, unsafe_allow_html=True)


def render_step3_complete_circuit(tokens: List[str]):
    """Render Step 3: Complete circuit overview."""

    st.markdown("""
    <div class="step-indicator">STEP 3: Complete Circuit</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### The Full Induction Circuit

    Now let's see how both heads work together via K-composition!
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### Circuit Composition:

        1. **Layer 0 - Previous Token Head**
           - Attends to previous position
           - Writes token[j-1] info → residual[j]

        2. **Layer 1 - Induction Head (K-composed)**
           - Computes K[j] from modified residual[j]
           - K[j] now encodes token[j-1]!
           - Q[i] matches K[j] when token[i] == token[j-1]
           - Attends to j (position *after* the match)
           - OV circuit copies → predict token[j]

        ### Mathematical View:

        The effective QK circuit implements:
        """)

        st.latex(r"\text{Attn}[i,j] \propto \text{Query}_i \cdot \text{Key}_j")
        st.latex(r"\approx W_{QK} \cdot \text{embed}(\text{token}_i) \cdot W_{OK}^{L0} \cdot \text{embed}(\text{token}_{j-1})")

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
        position i attends to position j because token[j-1] matches token[i].
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
                        f"because token[{src-1}] = '{tokens[src-1]}' matches token[{dest}] = '{tokens[dest]}'"
                    )
        else:
            st.info("No strong induction patterns detected. Try adding repeated words!")

    with analysis_tab[1]:
        st.markdown("### Previous Token Head Attention Pattern")
        st.markdown("""
        This shows the "attend to previous position" pattern that sets up
        K-composition for the induction mechanism.
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
                if src < len(tokens):
                    predicted = tokens[src]
                    st.markdown(f"""
                    <div class="concept-box">
                    At position <b>{dest}</b>, seeing "<b>{tokens[dest]}</b>":
                    <br><br>
                    • Found matching pattern: token[{src-1}] = "{tokens[src-1]}" = token[{dest}]
                    <br>
                    • Attends to position {src}: "{tokens[src]}"
                    <br>
                    • <b>Prediction: "{predicted}"</b>
                    </div>
                    """, unsafe_allow_html=True)


def render_generalisation():
    """Render the generalisation section - how to apply this to other circuits."""
    
    st.markdown("## 🎯 Generalising Beyond Induction Heads")
    
    st.markdown("""
    The induction head is just **one example** of a circuit. Here's the general 
    framework for finding and understanding circuits in any transformer.
    """)
    
    st.markdown("### The Circuit Discovery Recipe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="concept-box">
        <h4>Step 1: Identify a Behaviour</h4>
        <p>What computation does the model perform?</p>
        <ul>
        <li>Induction: [A][B]...[A] → [B]</li>
        <li>IOI: "Mary gave X to John" → John</li>
        <li>Greater-than: "1985 is after 19__" → ≥85</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="concept-box">
        <h4>Step 2: Localise Components</h4>
        <p>Which attention heads are responsible?</p>
        <ul>
        <li>Activation patching</li>
        <li>Ablation studies</li>
        <li>Attention pattern analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="concept-box">
        <h4>Step 3: Understand the Mechanism</h4>
        <p>For each head, ask:</p>
        <ul>
        <li>QK circuit: Where does it attend?</li>
        <li>OV circuit: What does it output?</li>
        <li>Composition: How do heads interact?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="concept-box">
        <h4>Step 4: Verify with Interventions</h4>
        <p>Test your hypothesis:</p>
        <ul>
        <li>Does ablating break the behaviour?</li>
        <li>Does it work on novel inputs?</li>
        <li>Can you predict failures?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Other Known Circuits")
    
    st.markdown("""
    | Circuit | Task | Components | Key Mechanism |
    |---------|------|------------|---------------|
    | **Induction** | [A][B]...[A] → [B] | prev_token + induction heads | K-composition |
    | **IOI** | "Mary gave to John" → John | ~26 heads | Name movers + inhibition |
    | **Greater-Than** | "1985 is after 19__" | Comparison heads | Magnitude detection |
    | **Docstring** | Complete Python docstrings | Argument copiers | Pattern matching |
    | **Copying** | Repeat salient tokens | Single heads | Positive OV eigenvalues |
    """)
    
    st.markdown("""
    <div class="insight-box">
    <b>The Key Insight:</b> All these circuits follow the same pattern — 
    they're subsets of model paths that implement a recognisable algorithm. 
    The path expansion framework applies to all of them!
    </div>
    """, unsafe_allow_html=True)


def render_advanced_details():
    """Render advanced technical details section."""

    st.markdown("## 🔬 Advanced: OV and QK Circuits in Detail")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### QK Circuit (What to attend to)

        The QK circuit determines the attention pattern:

        ```
        Attention[i,j] = softmax(Q[i] · K[j]^T / √d)
        ```

        **For Induction Heads (with K-composition):**

        The effective QK circuit matches:
        - Q encodes the current token's identity
        - K encodes the *previous* token's identity (via L0 composition)
        - High dot product when token[i] == token[j-1]

        **TransformerLens access:**
        ```python
        W_QK = model.W_Q[layer, head] @ model.W_K[layer, head].T
        # Shape: (d_model, d_model)
        
        # For K-composition analysis:
        W_K_composed = model.W_K[1, ind_head] @ model.W_O[0, prev_head]
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

        The OV circuit is a "copying" matrix:
        - Positive eigenvalues → increases logit of attended token
        - When attending to position j, boost P(token[j])

        **TransformerLens access:**
        ```python
        W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
        # Shape: (d_model, d_model)
        
        # Check if copying:
        eigenvalues = np.linalg.eigvals(W_OV)
        is_copying = (eigenvalues.real > 0).mean() > 0.7
        ```
        """)

    st.markdown("---")

    st.markdown("""
    ### 🧪 Verify Induction Heads in Real Models

    You can detect induction heads using the **prefix matching score**:

    ```python
    import transformer_lens as tl
    import torch

    # Load a model
    model = tl.HookedTransformer.from_pretrained("gpt2-small")

    # Create a repeated random sequence: [A][B][C]...[A][B][C]...
    seq_len = 50
    tokens = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
    repeated = torch.cat([tokens, tokens], dim=1)

    # Run with cache
    _, cache = model.run_with_cache(repeated)

    # Check attention patterns for induction signature
    for layer in range(model.cfg.n_layers):
        pattern = cache['pattern', layer][0]  # (n_heads, seq, seq)
        
        for head in range(model.cfg.n_heads):
            # Induction heads attend to position i - seq_len + 1
            # (the corresponding position in the first half)
            induction_stripe = pattern[head].diagonal(offset=-(seq_len-1))
            induction_score = induction_stripe.mean().item()
            
            if induction_score > 0.5:
                print(f"L{layer}H{head}: Induction score = {induction_score:.3f}")
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Measuring Composition Strength
    
    To find which heads compose, measure overlap in the residual stream subspaces:
    
    ```python
    def composition_score(W_out, W_in):
        '''
        Measure how much head_out's output subspace overlaps 
        with head_in's input subspace.
        '''
        # Frobenius norm of composition vs product of norms
        composed = W_in @ W_out
        score = (
            np.linalg.norm(composed, 'fro') / 
            (np.linalg.norm(W_in, 'fro') * np.linalg.norm(W_out, 'fro'))
        )
        return score
    
    # K-composition: does head0's output affect head1's keys?
    k_comp = composition_score(
        model.W_O[0, head0],
        model.W_K[1, head1]
    )
    
    # Q-composition: does head0's output affect head1's queries?
    q_comp = composition_score(
        model.W_O[0, head0],
        model.W_Q[1, head1]
    )
    
    # V-composition: does head0's output affect head1's values?
    v_comp = composition_score(
        model.W_O[0, head0],
        model.W_V[1, head1]
    )
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
    # 🔍 Induction Heads & Circuits Explorer
    ### Understanding Transformer Circuits Through Interactive Visualization
    """)

    st.markdown("---")

    # Navigation tabs
    tabs = st.tabs([
        "📖 Introduction",
        "🧠 What Is a Circuit?",
        "🔗 Composition Types",
        "🔧 Induction Step-by-Step",
        "🎮 Playground",
        "🎯 Generalisation",
        "🔬 Advanced Details"
    ])

    with tabs[0]:
        render_introduction()

    with tabs[1]:
        render_circuit_concept()

    with tabs[2]:
        render_composition_types()

    with tabs[3]:
        render_circuit_simulation()

    with tabs[4]:
        render_playground()

    with tabs[5]:
        render_generalisation()

    with tabs[6]:
        render_advanced_details()
        render_eigenvalue_analysis()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    Built with ❤️ for mechanistic interpretability learners<br>
    Terminology follows <a href="https://github.com/neelnanda-io/TransformerLens">TransformerLens</a> conventions<br>
    <br>
    <b>Key References:</b><br>
    <a href="https://transformer-circuits.pub/2021/framework/index.html">A Mathematical Framework for Transformer Circuits</a><br>
    <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html">In-context Learning and Induction Heads</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()