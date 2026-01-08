import logging
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from captum.attr import DeepLift
from scipy.linalg import svd

logger = logging.getLogger(__name__)


def _abs_attribution_func(multipliers, inputs, baselines):
    "Multiplier x abs(inputs - baselines) to avoid double-sign effects."
    # print(f"inputs: {inputs[1][:5]}")
    # print(f"baselines: {baselines[1][:5]}")
    # print(f"multipliers: {multipliers[0][:5]}")
    # print(f"multipliers: {multipliers[1][:5]}")
    return tuple(
        (input_ - baseline).abs() * multiplier
        for input_, baseline, multiplier in zip(inputs, baselines, multipliers)
    )


class TPCAV(torch.nn.Module):
    """End-to-end PCA fitting, projection, and attribution utilities."""

    def __init__(
        self,
        model,
        device: Optional[str] = None,
        layer_name: Optional[str] = None,
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
        self.layer_name = layer_name

    def list_module_names(self) -> List[str]:
        """List all module names in the model for layer selection."""
        return [name for name, _ in self.model.named_modules()]

    def tpcav_state_dict(self) -> Dict:
        """Export TPCAV buffers."""
        return {
            "layer_name": self.layer_name,
            "zscore_mean": getattr(self, "zscore_mean", None),
            "zscore_std": getattr(self, "zscore_std", None),
            "pca_inv": getattr(self, "pca_inv", None),
            "orig_shape": getattr(self, "orig_shape", None),
        }

    def restore_tpcav_state(self, tpcav_state_dict: Dict) -> None:
        """Load PCA buffers from disk."""
        self.layer_name = tpcav_state_dict["layer_name"]
        self._set_buffer("zscore_mean", tpcav_state_dict["zscore_mean"])
        self._set_buffer("zscore_std", tpcav_state_dict["zscore_std"])
        self._set_buffer("pca_inv", tpcav_state_dict["pca_inv"])
        self._set_buffer("orig_shape", tpcav_state_dict["orig_shape"])
        self.fitted = True
        logger.warning(
            "Restored TPCAV state, please set model attribute!\n\n Example: self.model = Model_class()",
        )

    def fit_pca(
        self,
        concepts: Iterable,
        num_samples_per_concept: int = 10,
        num_pc: Optional[int] | str = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample activations, compute PCA, and attach buffers to the model."""
        sampled_avs = []
        for concept in concepts:
            avs = self._sample_concept(concept, num_samples=num_samples_per_concept)
            logger.info(
                "Sampled %s activations from concept %s", avs.shape[0], concept.name
            )
            sampled_avs.append(avs)
        all_avs = torch.cat(sampled_avs)
        orig_shape = all_avs.shape
        flat = all_avs.flatten(start_dim=1)

        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        std[std == 0] = -1
        standardized = (flat - mean) / std

        v_inverse = None
        if num_pc is None or num_pc == "full":
            _, _, v = svd(standardized, lapack_driver="gesvd", full_matrices=False)
            v_inverse = torch.tensor(v)
        elif int(num_pc) == 0:
            v_inverse = None
        else:
            _, _, v = svd(standardized, lapack_driver="gesvd", full_matrices=False)
            v_inverse = torch.tensor(v[: int(num_pc)])

        self._set_buffer("zscore_mean", mean.to(self.device))
        self._set_buffer("zscore_std", std.to(self.device))
        self._set_buffer(
            "pca_inv", v_inverse.to(self.device) if v_inverse is not None else None
        )
        self._set_buffer("orig_shape", torch.tensor(orig_shape).to(self.device))
        self.fitted = True

        return {
            "zscore_mean": mean,
            "zscore_std": std,
            "pca_inv": v_inverse,
            "orig_shape": torch.tensor(orig_shape),
        }

    def project_activations(
        self, activations: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project flattened activations into PCA space and residual."""
        if not self.fitted:
            raise RuntimeError("Call fit_pca before projecting activations.")

        y = activations.flatten(start_dim=1).to(self.device)
        if self.pca_inv is not None:
            V = self.pca_inv.T
            zscore_mean = getattr(self, "zscore_mean", 0.0)
            zscore_std = getattr(self, "zscore_std", 1.0)
            y_standardized = (y - zscore_mean) / zscore_std
            y_projected = torch.matmul(y_standardized, V)
            y_residual = y_standardized - torch.matmul(y_projected, self.pca_inv)
            return y_residual, y_projected
        else:
            return y, None

    def concept_embeddings(self, concept, num_samples: int) -> torch.Tensor:
        """Return concatenated projected + residual activations for a concept."""
        avs = self._sample_concept(concept, num_samples=num_samples)
        residual, projected = self.project_activations(avs)
        if projected is not None:
            return torch.cat((projected, residual), dim=1)
        return residual

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
        name = self.layer_name
        if name is None:
            raise ValueError("layer name must be defined")
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
    ) -> Dict[str, torch.Tensor]:
        """Compute DeepLift attributions on PCA embedding space.

        target_batches and baseline_batches should yield (seq, chrom) pairs of matching length.
        """
        if not self.fitted:
            raise RuntimeError("Call fit_pca before attributing.")
        self.forward = self.forward_from_embeddings_at_layer
        deeplift = DeepLift(self, multiply_by_inputs=multiply_by_inputs)

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
                    additional_forward_args=(inputs,),
                    custom_attribution_func=(
                        None if not multiply_by_inputs else _abs_attribution_func
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
                        inputs,
                    ),
                    custom_attribution_func=(
                        None if not multiply_by_inputs else _abs_attribution_func
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

        return {
            "attributions": torch.cat(attributions),
        }

    def input_attributions(
        self,
        target_batches: Iterable,
        baseline_batches: Iterable,
        multiply_by_inputs: bool = True,
        cavs_list: List[torch.Tensor] | None = None,
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

    def _layer_output(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """Return activations from the configured layer or model hook."""
        if self.layer_name is None:
            # No layer configured; return model output directly.
            self.model(*inputs)
        layer = self._resolve_layer(self.layer_name)
        cache: List[torch.Tensor] = []

        def hook_fn(_module, _inputs, output):
            cache.append(output)

        handle = layer.register_forward_hook(hook_fn)
        try:
            inputs = [inp.to(self.device) if inp is not None else inp for inp in inputs]
            _ = self.model(*inputs)
        finally:
            handle.remove()

        if not cache:
            raise RuntimeError(f"No activation captured for layer {self.layer_name}")
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
        y_hat = torch.matmul(avs_projected, self.pca_inv) + avs_residual
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
        model_inputs: Tuple[torch.Tensor, Optional[torch.Tensor]],
        layer_activation: Optional[torch.Tensor] = None,
        cavs_list: Optional[List[torch.Tensor]] = None,
        mute_x_avs: bool = False,
        mute_remainder: bool = True,
    ) -> torch.Tensor:
        """
        Full forward pass with optional activation replacement and/or CAV-based gradient muting.
        """
        name = self.layer_name
        if name is None:
            raise ValueError("layer_name must be set on TPCAV to use forward_patched.")
        layer = self._resolve_layer(name)

        def hook_fn(_module, _inputs, output):
            y = layer_activation if layer_activation is not None else output
            if cavs_list is None or len(cavs_list) == 0:
                return y
            return self._disentangle_with_cavs(
                y, cavs_list, mute_x_avs=mute_x_avs, mute_remainder=mute_remainder
            )

        handle = layer.register_forward_hook(hook_fn)
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
