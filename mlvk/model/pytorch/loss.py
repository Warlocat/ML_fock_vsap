"""Loss terms used in the reported molecular ML-Fock calculations."""

import torch

from mlvk.model.pytorch.base import TorchBase


class VKLoss(TorchBase):
    """Matrix loss with optional MO-energy, density, and dipole penalties."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta_mo = kwargs.get("beta_mo", 0.0)
        self.beta_dm = kwargs.get("beta_dm", 0.0)
        self.beta_dipole = kwargs.get("beta_dipole", 0.0)
        self.scale = kwargs.get("scale", 1.0)
        self.idx_fock = kwargs.get("idx_fock", 0)
        self.mo_window_orb = kwargs.get("mo_window_orb")

        mo_window = kwargs.get("mo_window")
        if mo_window is None:
            self.mo_window = None
        else:
            self.register_buffer("mo_window", torch.tensor(mo_window) / self.scale)

    def _fock_penalty(self, fock_ml, mo_energy_true, eigh_X=None, nocc=None):
        if eigh_X is None:
            mo_energy_ml, mo_coeff_ml = torch.linalg.eigh(fock_ml)
        else:
            transformed = eigh_X @ fock_ml @ eigh_X.T
            mo_energy_ml, mo_coeff_ml = torch.linalg.eigh(transformed)

        if self.mo_window_orb is not None and nocc is not None:
            n_occ, n_virt = map(int, self.mo_window_orb)
            indices = torch.arange(mo_energy_true.shape[0], device=mo_energy_true.device)
            mask = (indices >= int(nocc) - n_occ) & (indices < int(nocc) + n_virt)
        elif self.mo_window is not None:
            mask = (mo_energy_true >= self.mo_window[0]) & (mo_energy_true <= self.mo_window[1])
        else:
            mask = torch.ones_like(mo_energy_true, dtype=torch.bool)

        loss = self.beta_mo * torch.mean((mo_energy_ml[mask] - mo_energy_true[mask]) ** 2)
        return loss, mo_coeff_ml

    @staticmethod
    def _dipole_penalty(dm_ml, dm_true, dipoles):
        dm_diff = dm_ml - dm_true
        return sum(torch.einsum("ij,ij->", dm_diff, component) ** 2 for component in dipoles)

    def forward(
        self,
        v_pred,
        v_true,
        mo_energy=None,
        dm=None,
        dipoles=None,
        prev_fock=None,
        eigh_X=None,
        nocc=None,
        do_extra_penalties=True,
        **_,
    ):
        loss = torch.mean((v_pred - v_true) ** 2)
        use_orbital_loss = self.beta_mo > 0 or self.beta_dm > 0 or self.beta_dipole > 0
        if not (do_extra_penalties and use_orbital_loss):
            return loss

        if mo_energy is None or nocc is None:
            raise ValueError("mo_energy and nocc are required by the configured orbital losses")

        fock_pred = v_pred[:, :, self.idx_fock]
        if prev_fock is not None:
            fock_pred = prev_fock + fock_pred
        mo_loss, mo_coeff_ml = self._fock_penalty(
            fock_pred, mo_energy, eigh_X=eigh_X, nocc=nocc
        )
        loss = loss + mo_loss

        if self.beta_dm > 0 or self.beta_dipole > 0:
            if dm is None:
                raise ValueError("dm is required by the configured density-based losses")
            mo_occ_ml = mo_coeff_ml[:, :nocc]
            dm_ml = 2.0 * mo_occ_ml @ mo_occ_ml.T
            if self.beta_dm > 0:
                loss = loss + self.beta_dm * torch.mean((dm_ml - dm) ** 2)
            if self.beta_dipole > 0:
                if dipoles is None or any(component is None for component in dipoles):
                    raise ValueError("dipole matrices are required by beta_dipole")
                loss = loss + self.beta_dipole * self._dipole_penalty(dm_ml, dm, dipoles)
        return loss
