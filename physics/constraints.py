import torch
import torch.nn as nn
import yaml


class PhysicsConstraints(nn.Module):
    def __init__(self, config_path='/home/hailemicaelyimer/Music/pi_slm/configs/physics.yaml'):
        super().__init__()

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['physics']

        self.warmup_epochs  = cfg['warmup_epochs']
        self.ramp_epochs    = cfg['ramp_epochs']
        self.final_weight   = cfg['final_weight']
        self.cfg            = cfg['constraints']

    def get_weight(self, epoch):
        """
        Physics loss weight schedule.
        Epoch 0-29:  weight = 0.0  (model learns basic regression first)
        Epoch 30-49: weight ramps from 0 to final_weight
        Epoch 50+:   weight = final_weight (0.01)

        Why: if we apply physics pressure too early the model gets
        confused before it has learned anything useful.
        """
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            return progress * self.final_weight
        else:
            return self.final_weight

    def constraint1_energy_structure(self, pred_dg, pred_rmsd, binder_mask):
        """
        Constraint 1: Energy-Structure Correlation

        Physical law: stronger binding (more negative ΔG) should
        correlate with more stable structure (lower RMSD).

        We compute the correlation between predicted ΔG and predicted RMSD
        on binder sequences. The correlation should be negative.
        If it is positive the model is predicting physically impossible combinations.

        Penalty = ReLU(correlation + 0.3)
        We allow small positive correlations (threshold -0.3) before penalizing.
        """
        if not self.cfg['energy_structure']['enabled']:
            return torch.tensor(0.0)

        # Only compute on binder sequences
        if binder_mask.sum() < 2:
            return torch.tensor(0.0, device=pred_dg.device)

        dg_b   = pred_dg[binder_mask]
        rmsd_b = pred_rmsd[binder_mask]

        # Compute correlation
        dg_centered   = dg_b   - dg_b.mean()
        rmsd_centered = rmsd_b - rmsd_b.mean()

        numerator   = (dg_centered * rmsd_centered).mean()
        denominator = (dg_centered.std() * rmsd_centered.std()).clamp(min=1e-8)
        correlation = numerator / denominator

        # Penalize if correlation is more positive than -0.3
        # (correlation should be negative — binding and stability go together)
        penalty = torch.relu(correlation + 0.3)

        weight = self.cfg['energy_structure']['weight']
        return weight * penalty

    def constraint2_hbond_feasibility(self, pred_rmsd, pred_hbonds, binder_mask):
        """
        Constraint 2: H-bond Feasibility

        Physical law: when RMSD is high (complex unstable/dissociating)
        the number of hydrogen bonds must be low.

        We penalize predictions where H-bonds are high AND RMSD is high.

        Penalty = mean(ReLU(pred_hbonds - max_allowed_hbonds))
        where max_allowed_hbonds decreases as RMSD increases.
        """
        if not self.cfg['hbond_feasibility']['enabled']:
            return torch.tensor(0.0)

        if binder_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_rmsd.device)

        max_hb = self.cfg['hbond_feasibility']['max_hbonds']

        rmsd_b   = pred_rmsd[binder_mask]
        hbonds_b = pred_hbonds[binder_mask]

        # As RMSD increases from 0 to 10, max allowed H-bonds
        # decreases from max_hb to 0
        # clamp RMSD to [0, 10] range for this calculation
        rmsd_normalized = (rmsd_b / 10.0).clamp(0, 1)
        max_allowed = max_hb * (1.0 - rmsd_normalized)

        # Penalize predictions above the maximum allowed
        violation = torch.relu(hbonds_b - max_allowed)
        penalty   = (violation ** 2).mean()

        weight = self.cfg['hbond_feasibility']['weight']
        return weight * penalty

    def constraint3_thermodynamic(self, pred_dg, binder_mask):
        """
        Constraint 3: Thermodynamic Consistency

        Physical law: known binders must have negative ΔG.
        Positive ΔG means no binding — thermodynamically impossible
        for a sequence we know binds to RBD.

        Penalty = mean(ReLU(pred_dg)) on binder sequences
        ReLU means we only penalize positive predictions — negative ones are fine.
        """
        if not self.cfg['thermodynamic']['enabled']:
            return torch.tensor(0.0)

        if binder_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_dg.device)

        dg_binders = pred_dg[binder_mask]

        # Penalize positive ΔG predictions on known binders
        violation = torch.relu(dg_binders)
        penalty   = violation.mean()

        weight = self.cfg['thermodynamic']['weight']
        return weight * penalty

    def forward(self, predictions, binder_mask, epoch):
        """
        Compute total physics loss.

        Args:
            predictions: dict from model forward pass
                         keys: delta_g, rmsd, h_bonds, binder_logit
            binder_mask: boolean tensor — True for known binder sequences
            epoch: current training epoch (for weight schedule)

        Returns:
            dict with individual constraint losses and total
        """
        # Get physics weight for current epoch
        physics_weight = self.get_weight(epoch)

        # Denormalize predictions back to physical units
        # (constraints are defined in physical units, not normalized)
        from data.dataset import HBD2Dataset
        pred_dg_raw   = predictions['delta_g']   * HBD2Dataset.DG_STD   + HBD2Dataset.DG_MEAN
        pred_rmsd_raw = predictions['rmsd']       * HBD2Dataset.RMSD_STD + HBD2Dataset.RMSD_MEAN
        pred_hb_raw   = predictions['h_bonds']    # already in physical units

        # Compute each constraint
        c1 = self.constraint1_energy_structure(
            pred_dg_raw, pred_rmsd_raw, binder_mask
        )
        c2 = self.constraint2_hbond_feasibility(
            pred_rmsd_raw, pred_hb_raw, binder_mask
        )
        c3 = self.constraint3_thermodynamic(
            pred_dg_raw, binder_mask
        )

        total = physics_weight * (c1 + c2 + c3)

        return {
            'constraint1': c1.item(),
            'constraint2': c2.item(),
            'constraint3': c3.item(),
            'total':       total,
            'weight':      physics_weight,
        }


# ─────────────────────────────────────────────────────────────
# TEST THIS FILE STANDALONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append('..')

    print("Testing physics constraints...")

    physics = PhysicsConstraints(config_path='/home/hailemicaelyimer/Music/pi_slm/configs/physics.yaml')
