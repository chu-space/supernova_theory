# Supernova Theory — SNEC Research Project

Astrophysics investigations using the [SuperNova Explosion Code (SNEC)](https://stellarcollapse.org/SNEC.html).

## Quick Start

```bash
# Build SNEC
cd snec && make

# Run default 15 M☉ RSG baseline
mkdir -p Data && ./snec

# Plot bolometric light curve
cd .. && python3 scripts/plot_lightcurve.py
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/SNEC_SETUP.md](docs/SNEC_SETUP.md) | Installation and build verification |
| [docs/SNEC_PARAMETER_GUIDE.md](docs/SNEC_PARAMETER_GUIDE.md) | Full parameter reference + physics overview |
| [docs/WEEK1_NICKEL_STUDY.md](docs/WEEK1_NICKEL_STUDY.md) | Week 1 nickel mass sensitivity study |
| [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) | Project status and command log |

## Directory Layout

```
supernova_theory/
├── snec/                  # SNEC v1.01 source + executable
│   ├── profiles/          # 15Msol_RSG and test profiles
│   ├── models/            # Morozova 2015 model grid
│   └── runs/              # Parameter study outputs
├── scripts/               # Automation and analysis
├── results/               # Plots and metrics
└── docs/                  # Documentation
```

## Python Dependencies

```bash
pip install -r requirements.txt
```
