# Induction Heads & Transformer Circuits Explorer

An interactive Streamlit application for understanding **Induction Heads**, a key mechanism in transformer in-context learning.

## Overview

This project provides an accessible, interactive exploration of how transformer models learn to copy previous tokens through induction heads. It's designed to demystify the inner workings of transformers by breaking down the mechanisms of Query (Q), Key (K), and Value (V) vectors in the attention mechanism.

The application follows **TransformerLens** conventions and terminology for technical accuracy.

## Features

- **Interactive Visualizations**: Explore Q, K, V vectors and attention patterns with interactive Plotly visualizations
- **Step-by-Step Learning**: Guide through how induction heads work with visual demonstrations
- **Technical Depth**: Detailed explanations of transformer components with code examples
- **AuDHD-Friendly Design**: Carefully crafted interface with:
  - High contrast visuals
  - Reduced visual clutter
  - Clear typography and readable line spacing
  - Intuitive navigation

## Key Concepts Covered

### Induction Heads
The core mechanism where transformers learn in-context patterns by:
1. Recognizing when a token has appeared before
2. Copying the value from that previous position
3. Enabling in-context learning without explicit training

### Transformer Components
- **Query (Q)**: "What am I looking for?" - Encodes the current token's identity
- **Key (K)**: "What information do I contain?" - Advertises what each position offers
- **Value (V)**: "What information should I share?" - The actual content to be copied

### Attention Mechanisms
Understanding how attention weights determine which previous tokens influence the current token's representation.

## Installation

### Requirements
- Python 3.12+
- Dependencies: numpy, plotly, streamlit

### Setup

Using [uv](https://astral.sh/blog/uv/) (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run induction_heads_explorer.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
circuit-explainer/
├── induction_heads_explorer.py    # Main Streamlit application
├── README.md                       # This file
├── pyproject.toml                  # Project configuration
├── uv.lock                         # Dependency lock file
└── .python-version                 # Python version specification
```

## Learning Path

The application guides you through:
1. **Foundational Concepts**: What are transformers and attention?
2. **Component Deep Dive**: Understanding Q, K, V matrices
3. **Attention Mechanism**: How vectors interact to produce attention weights
4. **Induction Heads**: The specific mechanism for copying tokens
5. **Interactive Examples**: Hands-on exploration with sample sequences

## Technical Notes

- All computations use standard transformer notation
- Visualization uses Plotly for interactivity
- Custom CSS styling ensures accessibility and clarity
- Code follows TransformerLens conventions for consistency

## Learning Resources

The application includes:
- Conceptual explanations with visual metaphors
- Mathematical notation for technical details
- Python code examples showing tensor operations
- Interactive visualizations of abstract concepts

## Accessibility

This project prioritizes accessibility with:
- High contrast color schemes (dark mode)
- Larger, readable fonts with generous line spacing
- Clear visual hierarchy and organization
- Simplified navigation structure
- Reduced motion animations (pulsing effects only when necessary)

## Future Enhancements

Potential additions:
- Support for different model architectures
- Custom sequence input and analysis
- Head-specific attention visualization
- Export/sharing of visualizations
- More interactive experiments

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## References

- TransformerLens: https://github.com/neelnathwani/TransformerLens
- "Transformer Circuits Thread" on LessWrong
- Interpretability research on attention mechanisms
