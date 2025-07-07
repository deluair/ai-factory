# 🏭 AI Token Factory Economics Stack

A comprehensive economic simulation platform modeling the complete AI token production pipeline - from raw materials (silicon, electricity, water) to intelligence tokens and productivity.

## 🎯 Overview

The modern factory is an AI token factory. This project simulates the three-layer economic stack:

- **Neoclouds**: Convert power + silicon → GPU clusters (SLURM/Kubernetes/SSH)
- **Inference Providers**: Transform GPU clusters → AI tokens
- **Applications**: Convert tokens → user productivity

```
Raw Materials → Neoclouds → Inference APIs → Applications → Productivity
(Si, Power, H2O)    ↓           ↓             ↓            ↓
                GPU Clusters → Tokens → Intelligence → Value
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run economic simulation
python main.py

# Launch interactive dashboard
streamlit run run_dashboard.py
```

## 📊 Key Features

- **Multi-layer Economic Analysis**: Track margins, profits, efficiency across the entire stack
- **Realistic Data**: GPU specs (H100, A100, V100), pricing models, usage patterns
- **Interactive Dashboard**: Streamlit visualization with executive summaries
- **Advanced Analytics**: Optimization recommendations, market benchmarking
- **Value Flow Tracking**: $44M raw materials → $17.7B productivity value (399.3x multiplier)

## 📁 Project Structure

```
ai-factory/
├── models/           # Economic layer models (neoclouds, inference, applications)
├── data/            # Simulation data generation
├── analytics/       # Dashboard, reports, economic analysis
├── utils/           # Formatters, helpers, calculations
├── config/          # Simulation parameters
├── main.py         # Core simulation engine
└── run_dashboard.py # Dashboard launcher
```

## 💡 Key Insights

- **ROI**: Each dollar invested generates $53.30 of productivity value
- **Value Transformation**: 399.3x multiplier from raw materials to intelligence
- **Stack Economics**: Complete cost/revenue tracking through all layers
- **Optimization**: Data-driven recommendations for efficiency improvements

## 🔬 Use Cases

- **Investment Analysis**: Evaluate AI infrastructure ROI
- **Strategic Planning**: Optimize resource allocation
- **Market Research**: Understand AI economics and pricing
- **Academic Research**: Study AI token production economics

## 🤝 Contributing

Contributions welcome! Submit issues and pull requests.

## 📄 License

MIT License