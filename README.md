# PI-SLM: Physics-Informed Small Language Model for Antimicrobial Peptide Design

## Overview
PI-SLM is a physics-informed small language model designed for antimicrobial peptide (AMP) modeling and generation.  
The model integrates sequence representation learning with physicochemical constraints to predict:

- Binding affinity (ΔG)
- Structural deviation (RMSD)
- Hydrogen bond interactions

## Architecture
Input → Embedding → Physchem → Time → Transformer → Outputs

## Usage

python main.py --mode pretrain
python main.py --mode finetune
python main.py --mode evaluate
python main.py --mode all

## Structure
configs/
data/
model/
training/
evaluation/
results/

## Outputs
ΔG, RMSD, H-bonds
