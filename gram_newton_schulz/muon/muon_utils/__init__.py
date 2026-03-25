from .muon_opt_utils import (
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    muon_update_pre_orthogonalize,
    muon_update_post_orthogonalize,
    create_param_batches,
    get_or_initialize_muon_state,
)

from .muon_matrix_split_utils import (
    get_newton_schulz_inputs_from_gradients,
    scale_newton_schulz_outputs_with_adjusted_lr,
    reconstruct_update_from_newton_schulz_outputs,
)

__all__ = [
    # Optimizer utils
    "adjust_lr_rms_norm",
    "adjust_lr_spectral_norm",
    "muon_update_pre_orthogonalize",
    "muon_update_post_orthogonalize",
    "create_param_batches",
    "get_or_initialize_muon_state",
    # Matrix split utils
    "get_newton_schulz_inputs_from_gradients",
    "scale_newton_schulz_outputs_with_adjusted_lr",
    "reconstruct_update_from_newton_schulz_outputs",
]
