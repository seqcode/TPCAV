import logging
from functools import partial
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from captum.attr import DeepLift, LayerDeepLift
from scipy.linalg import svd

logger = logging.getLogger(__name__)


def _abs_attribution_func(multipliers, inputs, baselines):
    "Multiplier x abs(inputs - baselines) to avoid double-sign effects."
    return tuple(
        (input_ - baseline).abs() * multiplier
        for input_, baseline, multiplier in zip(inputs, baselines, multipliers)
    )

class Untracked:
    def __init__(self, module):
        self.module = module

class TPCAV(torch.nn.Module):
    """End-to-end PCA fitting, projection, and attribution utilities."""

    def __init__(
        self,
        model,
        device: Optional[str] = None,
        layer_name: Optional[str] = None,
        layer: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        layer_name: optional module name to intercept activations via forward hook
                    (useful if forward_until_select_layer is not implemented).
        """
        super().__init__()
        self.model = model
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.fitted = False
        self._full_transform = False
        self.fitting_mode: Optional[str] = None
        if layer is not None:
            self.layer = Untracked(layer)
            self.layer_name = ""
        elif layer_name is not None:
            self.layer = Untracked(self._resolve_layer(layer_name))
            self.layer_name = layer_name
        else:
            raise Exception(
                "You have to specify either layer or layer_name to construct TPCAV model"
            )

    def save_state(self, output_path="tpcav_state.pt"):
        """
        Save TPCAV model state to a file.
        This is useful when model could not be restored by torch.save/load
        """
        state = {
            "fitted": self.fitted,
            "zscore_mean": self.zscore_mean,
            "zscore_std": self.zscore_std,
            "Vh": self.Vh,
            "orig_shape": self.orig_shape
        }
        torch.save(state, output_path)
    
    @staticmethod
    def load_state(model, layer_name=None, 
                   layer=None, state_path: str = "tpcav_state.pt"):
        """
        Restore CavTrainer state from a file.
        """
        tpcav_model = TPCAV(model, layer_name=layer_name, layer=layer)

        state = torch.load(state_path, map_location="cpu")
        tpcav_model.fitted = state["fitted"]
        tpcav_model._set_buffer("zscore_mean", state["zscore_mean"].to(tpcav_model.device))
        tpcav_model._set_buffer("zscore_std", state["zscore_std"].to(tpcav_model.device))
        tpcav_model._set_buffer("Vh", state["Vh"].to(tpcav_model.device) if state["Vh"] is not None else None)
        tpcav_model._set_buffer("orig_shape", state["orig_shape"].to(tpcav_model.device))

        logger.info("Successfully restored tpcav model states!")
        return tpcav_model

    def list_module_names(self) -> List[str]:
        """List all module names in the model for layer selection."""
        return [name for name, _ in self.model.named_modules()]

    def _collect_concept_examples(self, concepts, num_samples_per_concept: int) -> torch.Tensor:
        """Collect concept examples from each concept and return as a single tensor."""
        sampled_avs = []
        for concept in concepts:
            avs = self._sample_concept(concept, num_samples=num_samples_per_concept)
            logger.info(
                "Sampled %s activations from concept %s", avs.shape[0], concept.name
            )
            sampled_avs.append(avs)
        return torch.cat(sampled_avs)

    def _evaluate_transformation(self, activations: torch.Tensor, num_dims_to_evaluate=500) -> None:
        """Evaluate the quality of the linear transformation by comparing the correlation between dimensions before and after transformation."""
        if not self.fitted:
            raise RuntimeError("Call fit_pca before evaluating transformation.")
        activations_flat = activations.flatten(start_dim=1)
        avs_residual, avs_projected = self.project_activations(activations_flat)

        # subsample dimensions for correlation evaluation if there are too many
        def compute_mean_abs_corr(tensor):
            if tensor.shape[1] > num_dims_to_evaluate:
                tensor = tensor[:, torch.randperm(tensor.shape[1])[:num_dims_to_evaluate]]
            # compute pearson correlation matrix and return mean absolute values in the matrix
            corr = torch.corrcoef(tensor.T) # [dims, dims]
            k = corr.shape[0]
            off_diag = corr.abs().sum() - k  # sum of absolute values minus diagonal
            return (off_diag / (k * (k - 1))).item()

        logger.info("Mean absolute pearson correlation on concept activations before transformation: %.4f", compute_mean_abs_corr(activations_flat))
        if not self._full_transform:
            logger.info("Mean absolute pearson correlation on residuals after transformation: %.4f", compute_mean_abs_corr(avs_residual))
        if avs_projected is not None:
            logger.info("Mean absolute pearson correlation on projected activations after transformation: %.4f", compute_mean_abs_corr(avs_projected))

    def fit_pca(
        self,
        concepts: Iterable,
        num_samples_per_concept: int = 10,
        num_pc: Optional[Union[int, str]] = None,
    ) -> None:
        """
        Sample activations from the provided concepts, compute PCA, and attach
        transformation buffers to the model.

        num_pc can be an integer specifying the number of principal components to keep,
        or "full" to keep all components. If num_pc is 0 or "none", no PCA projection
        will be applied and all activations will be treated as residuals.
        """
        logger.info("Start building PCA transformation.")

        all_avs = self._collect_concept_examples(concepts, num_samples_per_concept)
        orig_shape = all_avs.shape
        flat = all_avs.flatten(start_dim=1)

        logger.info("Computing PCA on %s samples with %s features", flat.shape[0], flat.shape[1])

        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        std[std == 0] = -1
        standardized = (flat - mean) / std

        if num_pc is None or num_pc == "full":
            _, S, Vh = svd(standardized, lapack_driver="gesvd", full_matrices=False)
            Vh = torch.tensor(Vh)
            self._full_transform = True
        elif int(num_pc) == 0:
            S = None
            Vh = None
            self._full_transform = False
        else:
            _, S, Vh = svd(standardized, lapack_driver="gesvd", full_matrices=False)
            self._full_transform = False if int(num_pc) < standardized.shape[1] else True
            Vh = torch.tensor(Vh[: int(num_pc)])

        self.eigen_values = np.square(S) if S is not None else None

        if S is not None:
            var_explained = np.square(S) / np.square(S).sum()
            top_n = min(10, len(var_explained))
            top_cumulative = var_explained[:top_n].cumsum()
            lines = "  ".join(
                f"PC{i+1}: {var_explained[i]*100:.1f}% (cum {top_cumulative[i]*100:.1f}%)"
                for i in range(top_n)
            )
            logger.info("Variance explained — %s", lines)

        self._set_buffer("zscore_mean", mean.to(self.device))
        self._set_buffer("zscore_std", std.to(self.device))
        self._set_buffer("Vh", Vh.to(self.device) if Vh is not None else None)
        self._set_buffer("orig_shape", torch.tensor(orig_shape).to(self.device))
        self.fitted = True
        self.fitting_mode = "pca"

        logger.info("PCA transformation built.")
        self._evaluate_transformation(all_avs)

    def fit_decorr(
        self,
        concepts: Iterable,
        num_samples_per_concept: int = 50,
        lr: float = 1e-3,
        max_epochs: int = 500,
        max_steps_per_epoch: int = 1000,
        patience: int = 20,
        lam_var: float = 1e-4,
        lam_cov: float = 1.0,
        num_dims_sample: Optional[int] = None,
        batch_size: int = 64,
        target_batches: Optional[Iterable] = None,
        baseline_batches: Optional[Iterable] = None,
        weight_power: float = 2.0,
        weight_floor: float = 1.0,
    ) -> None:
        """
        Alternative to fit_pca using a learned linear decorrelation layer.

        Collects concept activations then trains nn.Linear(D, D, bias=False) with a
        VICReg-style variance + covariance loss. Sets the same buffers as fit_pca so
        the rest of the pipeline is unaffected. Set num_dims_sample to subsample output
        dimensions when computing the covariance matrix, reducing GPU memory from
        O(D^2) to O(num_dims_sample^2).

        target_batches / baseline_batches: when both are provided, DeepLift attributions
            on the raw layer activations (via raw_layer_attributions) are used to derive
            per-dimension importance weights. weight_power amplifies contrast between
            dimensions (2.0 = square). weight_floor is added after normalization so every
            dimension retains a minimum weight, preventing low-attribution dims from being
            ignored entirely.
        """
        logger.info("fit_decorr: collecting concept activations.")
        all_avs = self._collect_concept_examples(concepts, num_samples_per_concept)
        orig_shape = all_avs.shape
        flat = all_avs.flatten(start_dim=1).float()
        N, D = flat.shape

        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        std[std == 0] = -1
        flat = (flat - mean) / std

        self._set_buffer("zscore_mean", mean.to(self.device))
        self._set_buffer("zscore_std",  std.to(self.device))

        # ── Dimension importance weights from DeepLift attributions ──────────
        dim_weights: Optional[torch.Tensor] = None
        if target_batches is not None and baseline_batches is not None:
            logger.info("fit_decorr: computing dimension importance weights from DeepLift attributions.")
            attrs = self.raw_layer_attributions(target_batches, baseline_batches)  # (N, D)
            importance = attrs.abs().mean(dim=0).to(self.device).pow(weight_power)
            dim_weights = (importance / importance.mean().clamp(min=1e-8) + weight_floor).detach()
            logger.info(
                "fit_decorr: dimension weights computed; max=%.3f, min=%.3f",
                dim_weights.max().item(), dim_weights.min().item(),
            )

        # ── Training ──────────────────────────────────────────────────────────
        logger.info("fit_decorr: training on %d samples, %d features.", N, D)

        proj = torch.nn.Linear(D, D, bias=False, device=self.device)
        torch.nn.init.orthogonal_(proj.weight)
        optimizer = torch.optim.AdamW(proj.parameters(), lr=lr)
        best_loss: Optional[float] = None
        best_weight = proj.weight.detach().clone()
        patience_count = 0

        for epoch in range(max_epochs):
            perm = torch.randperm(N)
            epoch_losses = []
            var_losses = []
            cov_losses = []

            for i in range(0, N, batch_size):
                batch = flat[perm[i:i + batch_size]].to(self.device)
                z     = proj(batch)
                n, _  = z.shape
                z_c   = z - z.mean(dim=0)

                if num_dims_sample is not None and num_dims_sample < D:
                    dim_idx = torch.randperm(D, device=self.device)[:num_dims_sample]
                    z_sub   = z_c[:, dim_idx]
                    k       = num_dims_sample
                    w_sub   = dim_weights[dim_idx] if dim_weights is not None else None
                else:
                    z_sub  = z_c
                    k      = D
                    w_sub  = dim_weights
                cov = (z_sub.T @ z_sub) / (n - 1)

                if w_sub is not None:
                    var_loss = (torch.relu(1 - cov.diagonal().sqrt()) * w_sub).mean()
                    cov_loss = (cov.pow(2).fill_diagonal_(0) * (w_sub[:, None] * w_sub[None, :])).sum() / k
                else:
                    var_loss = torch.relu(1 - cov.diagonal().sqrt()).mean()
                    cov_loss = cov.pow(2).fill_diagonal_(0).sum() / k
                loss = lam_var * var_loss + lam_cov * cov_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                var_losses.append(var_loss.item())
                cov_losses.append(cov_loss.item())

                if i > max_steps_per_epoch:
                    break

            epoch_loss = float(np.mean(epoch_losses))
            epoch_var_loss = float(np.mean(var_losses))
            epoch_cov_loss = float(np.mean(cov_losses))
            logger.info("fit_decorr epoch %d/%d: loss=%.4f, var_loss=%.4f, cov_loss=%.4f", epoch + 1, max_epochs, epoch_loss, epoch_var_loss, epoch_cov_loss)

            if best_loss is None or epoch_loss < best_loss:
                best_loss      = epoch_loss
                best_weight    = proj.weight.detach().clone()
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info("fit_decorr: early stopping at epoch %d", epoch + 1)
                    break

        self._set_buffer("Vh", best_weight)
        self._set_buffer("orig_shape",  torch.tensor(orig_shape).to(self.device))
        self.fitted = True
        self.fitting_mode = "decorr"
        logger.info("fit_decorr: done.")
        self._evaluate_transformation(all_avs)

    def project_activations(
        self, activations: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project flattened activations into PCA space and residual."""
        if not self.fitted:
            logger.warning(
                "PCA not fit before projecting activations, make sure this is intended"
            )

        y = activations.flatten(start_dim=1).to(self.device)
        if self.Vh is not None:
            V = self.Vh.T
            zscore_mean = getattr(self, "zscore_mean", 0.0)
            zscore_std = getattr(self, "zscore_std", 1.0)
            y_standardized = (y - zscore_mean) / zscore_std
            y_projected = torch.matmul(y_standardized, V)
            y_residual = y_standardized - torch.matmul(y_projected, self.Vh)
            return y_residual, y_projected
        else:
            return y, None

    def concept_embeddings(self, concept, num_samples: int) -> torch.Tensor:
        """Return concatenated projected + residual activations for a concept."""
        residual, projected = self._sample_concept_embeddings(concept, num_samples=num_samples)

        if projected is not None:
            return torch.cat((projected, residual), dim=1)
        return residual.detach()

    def forward_from_embeddings_at_layer(
        self,
        avs_residual: torch.Tensor,
        avs_projected: Optional[torch.Tensor] = None,
        model_inputs: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Resume model forward by injecting activations at a named layer.

        If layer_name is not set, falls back to forward_with_embeddings which
        expects model.forward_from_projected_and_residual to exist.
        """
        if model_inputs is None:
            raise ValueError(
                "model_inputs (seq, chrom) must be provided to run forward."
            )

        y_hat = self.embedding_to_layer_activation(avs_residual, avs_projected)
        return self.forward_patched(
            model_inputs,
            layer_activation=y_hat,
        )

    def layer_attributions(
        self,
        target_batches: Iterable,
        baseline_batches: Iterable,
        multiply_by_inputs: bool = True,
        abs_inputs_diff: bool = True,
    ) -> torch.Tensor:
        """
        Compute DeepLift attributions on PCA embedding space.

        By default, it computes (input - baseline).abs() * multiplier to avoid double-sign effects (abs_inputs_diff=True).

        target_batches and baseline_batches should yield (seq, chrom) pairs of matching length.
        """
        if not self.fitted:
            raise RuntimeError("Call fit_pca before attributing.")
        self.forward = self.forward_from_embeddings_at_layer
        deeplift = DeepLift(self, multiply_by_inputs=multiply_by_inputs)

        custom_attr_func = _abs_attribution_func if abs_inputs_diff else None

        attributions = []
        for inputs, binputs in zip(target_batches, baseline_batches):
            avs = self._layer_output(*[i.to(self.device) for i in inputs])
            avs_residual, avs_projected = self.project_activations(avs)

            bavs = self._layer_output(*[bi.to(self.device) for bi in binputs])
            bavs_residual, bavs_projected = self.project_activations(bavs)

            # detach the projected tensor as it's connnected to the original input graph,
            # detaching it would keep the gradients on it
            if avs_projected is not None:
                avs_projected = avs_projected.detach()
                bavs_projected = bavs_projected.detach()
                attribution = deeplift.attribute(
                    (avs_residual.to(self.device), avs_projected.to(self.device)),
                    baselines=(
                        bavs_residual.to(self.device),
                        bavs_projected.to(self.device),
                    ),
                    additional_forward_args=(
                        [torch.cat([i, bi]) for i, bi in zip(inputs, binputs)],
                    ),
                    custom_attribution_func=(
                        None if not multiply_by_inputs else custom_attr_func
                    ),
                )
                attr_residual, attr_projected = attribution
                attribution = torch.cat((attr_projected, attr_residual), dim=1)
            else:
                attribution = deeplift.attribute(
                    (avs_residual.to(self.device),),
                    baselines=(bavs_residual.to(self.device),),
                    additional_forward_args=(
                        None,
                        [torch.cat([i, bi]) for i, bi in zip(inputs, binputs)],
                    ),
                    custom_attribution_func=(
                        None if not multiply_by_inputs else custom_attr_func
                    ),
                )[0]

            attributions.append(attribution.detach().cpu())

            with torch.no_grad():
                del (
                    avs,
                    avs_projected,
                    avs_residual,
                    bavs,
                    bavs_projected,
                    bavs_residual,
                )
                torch.cuda.empty_cache()

        return torch.cat(attributions)

    def raw_layer_attributions(
        self,
        target_batches: Iterable,
        baseline_batches: Iterable,
        multiply_by_inputs: bool = True,
    ) -> torch.Tensor:
        """
        Compute DeepLift attributions directly on raw (un-projected) layer activations.

        Unlike layer_attributions, this does not require fit_pca or fit_decorr to have
        been called. Returns a tensor of shape (N, D) where D is the flattened layer
        output size.
        """
        self.forward = self.forward_patched_tensor
        deeplift = LayerDeepLift(self, multiply_by_inputs=multiply_by_inputs, layer=self.layer.module)

        attributions = []
        for inputs, binputs in zip(target_batches, baseline_batches):

            attribution = deeplift.attribute(
                tuple([i.to(self.device) for i in inputs]),
                baselines=tuple([bi.to(self.device) for bi in binputs]),
            )
            attributions.append(attribution.detach().cpu())

            with torch.no_grad():
                del inputs, binputs
                torch.cuda.empty_cache()

        return torch.cat(attributions).flatten(start_dim=1)

    def input_attributions(
        self,
        target_batches: Iterable,
        baseline_batches: Iterable,
        multiply_by_inputs: bool = True,
        cavs_list: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Compute DeepLift attributions on PCA embedding space.

        target_batches and baseline_batches should yield (seq, chrom) pairs of matching length.
        """
        if not self.fitted:
            raise RuntimeError("Call fit_pca before attributing.")
        self.forward = partial(
            self.forward_patched_tensor,
            layer_activation=None,
            cavs_list=cavs_list,
            mute_x_avs=False,
            mute_remainder=True,
        )
        deeplift = DeepLift(self, multiply_by_inputs=multiply_by_inputs)

        attributions = []
        for inputs, binputs in zip(target_batches, baseline_batches):
            attribution = deeplift.attribute(
                tuple([i.to(self.device) for i in inputs]),
                baselines=tuple([bi.to(self.device) for bi in binputs]),
            )
            attributions.append(
                [a.detach().cpu() for a in attribution]
                if isinstance(attribution, tuple)
                else attribution.detach().cpu()
            )

        return [torch.cat(z) for z in zip(*attributions)]

    def _sample_concept(self, concept, num_samples: int) -> torch.Tensor:
        avs: List[torch.Tensor] = []
        num = 0
        for inputs in concept.data_iter:
            av = self._layer_output(*[i.to(self.device) for i in inputs])
            avs.append(av.detach().cpu())
            num += av.shape[0]
            if num >= num_samples:
                break
        if not avs:
            raise ValueError(f"No activations gathered for concept {concept.name}")
        return torch.cat(avs)[:num_samples]

    def _sample_concept_embeddings(self, concept, num_samples: int) -> List[Union[torch.Tensor, None]]:
        num = 0
        residuals = []; projected = []
        for inputs in concept.data_iter:
            av = self._layer_output(*[i.to(self.device) for i in inputs]).detach()
            r, p = self.project_activations(av)
            residuals.append(r.cpu())
            if p is not None:
                projected.append(p.cpu())
            else:
                projected.append(p)
            num += av.shape[0]
            if num >= num_samples:
                break
        if not residuals:
            raise ValueError(f"No activations gathered for concept {concept.name}")
        residuals = torch.cat(residuals)
        projected = torch.cat(projected) if p is not None else None
        return residuals, projected

    def _layer_output(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """Return activations from the configured layer or model hook."""
        cache: List[torch.Tensor] = []

        def hook_fn(_module, _inputs, output):
            cache.append(output)

        handle = self.layer.module.register_forward_hook(hook_fn)
        try:
            inputs = [inp.to(self.device) if inp is not None else inp for inp in inputs]
            _ = self.model(*inputs)
        finally:
            handle.remove()

        if not cache:
            raise RuntimeError(f"No activation captured for layer {self.layer_name}; Are you sure it's used in forward function?")
        return cache[0]

    def _resolve_layer(self, name: str):
        for module_name, module in self.model.named_modules():
            if module_name == name:
                return module
        raise ValueError(f"Layer {name} not found in model.")

    def embedding_to_layer_activation(
        self, avs_residual: torch.Tensor, avs_projected: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Combine projected/residual embeddings into the layer activation space,
        mirroring scripts/models.py merge logic.
        """
        if self.Vh is not None:
            y_hat = torch.matmul(avs_projected, self.Vh) + avs_residual
        else:
            y_hat = avs_residual
        y_hat = y_hat * self.zscore_std + self.zscore_mean

        return y_hat.reshape((-1, *self.orig_shape[1:]))

    def forward_patched_tensor(
        self,
        *model_inputs: torch.Tensor,
        layer_activation: Optional[torch.Tensor] = None,
        cavs_list: Optional[List[torch.Tensor]] = None,
        mute_x_avs: bool = False,
        mute_remainder: bool = True,
    ) -> torch.Tensor:
        return self.forward_patched(
            model_inputs, layer_activation, cavs_list, mute_x_avs, mute_remainder
        )

    def forward_patched(
        self,
        model_inputs: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        layer_activation: Optional[torch.Tensor] = None,
        cavs_list: Optional[List[torch.Tensor]] = None,
        mute_x_avs: bool = False,
        mute_remainder: bool = True,
    ) -> torch.Tensor:
        """
        Full forward pass with optional activation replacement and/or CAV-based gradient muting.
        """

        def hook_fn(_module, _inputs, output):
            y = layer_activation if layer_activation is not None else output
            if cavs_list is None or len(cavs_list) == 0:
                return y
            return self._disentangle_with_cavs(
                y, cavs_list, mute_x_avs=mute_x_avs, mute_remainder=mute_remainder
            )

        handle = self.layer.module.register_forward_hook(hook_fn)
        try:
            output = self.model(*[i.to(self.device) for i in model_inputs])
        finally:
            handle.remove()
        return output

    def _disentangle_with_cavs(
        self,
        layer_output: torch.Tensor,
        cavs_list: List[torch.Tensor],
        mute_x_avs: bool = False,
        mute_remainder: bool = True,
    ) -> torch.Tensor:
        """
        Project activations into CAV subspace and optionally zero gradients for
        subspaces (default mutes orthogonal/remainder gradients).
        """
        y = layer_output.flatten(start_dim=1)
        y_residual, y_projected = self.project_activations(y)
        y_pca_all = torch.cat([y_projected, y_residual], dim=1)

        cavs_matrix = torch.stack(cavs_list, dim=1).to(y.device)  # [dims, #cavs]

        if cavs_matrix.shape[1] > cavs_matrix.shape[0]:
            logger.warning(
                "CAV matrix has more columns than rows; remainder should be near zero."
            )

        mrank = torch.linalg.matrix_rank(cavs_matrix)
        cavs_ortho = torch.linalg.qr(cavs_matrix, mode="reduced").Q[:, :mrank].detach()
        if not torch.allclose(
            cavs_ortho.T @ cavs_ortho,
            torch.eye(mrank, device=cavs_ortho.device),
            atol=1e-3,
            rtol=1e-3,
        ):
            logger.warning("Q^TQ not identity; check CAV matrix conditioning.")

        y_pca_x_avs = y_pca_all @ cavs_ortho @ cavs_ortho.T
        y_pca_remainder = y_pca_all - y_pca_x_avs

        if mute_x_avs:
            y_pca_x_avs.register_hook(lambda grad: torch.zeros_like(grad))
        if mute_remainder:
            y_pca_remainder.register_hook(lambda grad: torch.zeros_like(grad))

        y_pca_hat = y_pca_x_avs + y_pca_remainder

        dim_projected = y_projected.shape[1] if y_projected is not None else 0

        return self.embedding_to_layer_activation(
            y_pca_hat[:, dim_projected:], y_pca_hat[:, :dim_projected]
        )

    def _set_buffer(self, name: str, value: Optional[torch.Tensor]) -> None:
        if hasattr(self.model, "_buffers") and name in self.model._buffers:
            self._buffers[name] = value  # type: ignore[index]
        else:
            self.register_buffer(name, value)
