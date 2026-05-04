"""
main.py

Single entry point for the PI-SLM pipeline.

Usage:
    python main.py --mode pretrain
    python main.py --mode finetune
    python main.py --mode evaluate
    python main.py --mode all
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def run_pretrain():
    print("\n" + "=" * 70)
    print("STAGE 1 — PRETRAINING ON DBAASP")
    print("=" * 70)
    from pretrain.pretrain import pretrain
    pretrain(
        config_path  = '/home/hailemicaelyimer/Music/pi_slm/configs/training.yaml',
        model_config = '/home/hailemicaelyimer/Music/pi_slm/configs/model.yaml',
        dbaasp_csv   = '/home/hailemicaelyimer/Music/pi_slm/data/raw/dbaasp/dbaasp_pretrain.csv',
        save_path    = '/home/hailemicaelyimer/Music/pi_slm/results/checkpoints/pretrained_model.pt',
    )


def run_finetune():
    print("\n" + "=" * 70)
    print("STAGE 2 — FINE-TUNING ON hBD-2 DATA")
    print("=" * 70)
    from training.trainer import train
    train(
        config_path     = 'configs/training.yaml',
        model_config    = 'configs/model.yaml',
        physics_config  = 'configs/physics.yaml',
        pretrained_path = 'results/checkpoints/pretrained_model.pt',
        excel_path      = 'data/raw/consolidated_cleaned.xlsx',
    )
def run_evaluate():
    print("\n" + "=" * 70)
    print("EVALUATION + SEQUENCE GENERATION")
    print("=" * 70)
    from evaluation.evaluate import evaluate
    evaluate(
        checkpoint_path = 'results/checkpoints/best_model.pt',
        model_config    = 'configs/model.yaml',
        config_path     = 'configs/training.yaml',
        generate        = True,
        n_generate      = 30000,
        top_k           = 30,
        save_results    = True,
    )


def main():
    parser = argparse.ArgumentParser(description='PI-SLM Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['pretrain', 'finetune', 'evaluate', 'all'],
        help='Which stage to run'
    )
    args = parser.parse_args()

    if args.mode == 'pretrain':
        run_pretrain()

    elif args.mode == 'finetune':
        run_finetune()

    elif args.mode == 'evaluate':
        run_evaluate()

    elif args.mode == 'all':
        run_pretrain()
        run_finetune()
        run_evaluate()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()