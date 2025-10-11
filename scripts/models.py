Description = "NN model collection"

import torch


class Model_Class(torch.nn.Module):
    """
    Model definition
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def merge_projected_and_residual(self, y_projected, y_residual):
        "Project PCA activations back to original space and add with residual, then multiply with zscore_std"
        "NOTE: zscore_mean is not added here , this is to avoid adding it twice if you disentange y_projected and y_residual"
        y_hat = torch.matmul(y_projected, self.pca_inv) + y_residual  # PCA inverse
        y_hat = y_hat * self.zscore_std  # reverse normalize

        return y_hat

    def forward_until_select_layer(self, seq, chrom):

        return y_hat

    def resume_forward_from_select_layer(self, y_hat):
        return y_pred

    def forward_from_start(
        self,
        seq,
        chrom,
        cavs_list=None,
        mute_x_avs=False,
        mute_remainder=False,
        project_to_pca=True,
    ):
        y_hat = self.forward_until_select_layer(seq, chrom)

        if project_to_pca:
            avs_residual, avs_projected = self.project_post_attn_to_pca(
                y_hat.flatten(start_dim=1)
            )

            return self.forward_from_projected_and_residual(
                avs_residual,
                avs_projected,
                cavs_list,
                mute_x_avs,
                mute_remainder,
            )
        else:
            return self.resume_forward_from_select_layer(y_hat)

    def forward_from_projected_and_residual(
        self,
        y_residual,
        y_projected,
        cavs_list=None,
        mute_x_avs=False,
        mute_remainder=False,
    ):
        if cavs_list is not None:
            y_hat_x_avs, y_hat_remainder = self.disentangle_avs_x_cavs(
                y_projected, y_residual, cavs_list, mute_x_avs, mute_remainder
            )

            if mute_x_avs:  # which part of the activations to use
                y_hat_x_avs.register_hook(lambda grad: torch.zeros_like(grad))
            if mute_remainder:
                y_hat_remainder.register_hook(lambda grad: torch.zeros_like(grad))
            y_hat = y_hat_x_avs + y_hat_remainder + self.zscore_mean
        else:
            y_hat = (
                self.merge_projected_and_residual(y_projected, y_residual)
                + self.zscore_mean
            )

        if self.orig_shape is not None:
            y_hat = y_hat.reshape((-1, *self.orig_shape[1:]))

        # resume back to normal forward process
        y_hat = self.resume_forward_from_select_layer(y_hat)
        return y_hat

    def project_avs_to_pca(self, y):
        return self.project_post_attn_to_pca(y)

    def project_post_attn_to_pca(self, y):
        "Project activations of post attn layer to PCA space"
        if self.pca_inv is not None:
            V = self.pca_inv.T
            y_standardized = (y - self.zscore_mean) / self.zscore_std
            y_projected = torch.matmul(y_standardized, V)

            y_residual = y_standardized - torch.matmul(y_projected, self.pca_inv)

            return y_residual, y_projected
        else:
            return y, None

    def disentangle_avs_x_cavs(
        self, y_projected, y_residual, cavs_list, mute_x_avs=False, mute_remainder=False
    ):
        "Given a list of CAVs, disentangle the activations"
        y_all = torch.cat(
            [y_projected, y_residual], dim=1
        )  # merge projected and residual activations because cavs is computed on projected + residual

        cavs_matrix = torch.stack(cavs_list, dim=1).to(y_all.device)  # [#dims, #cavs]

        if cavs_matrix.shape[1] > cavs_matrix.shape[0]:
            print(
                f"Warning: CAVs matrix has more CAVs than dimensions! Remainder should be super close to 0"
            )

        # check the rank of cavs_matrix first
        mrank = torch.linalg.matrix_rank(cavs_matrix)

        cavs_ortho_matrix = (
            torch.linalg.qr(cavs_matrix, mode="reduced").Q[:, :mrank].detach()
        )  # [#dims, #cavs], then keep the first mrank orthogonal basis as the remaining ones should be close to 0 and meaningless
        assert torch.allclose(
            cavs_ortho_matrix.T @ cavs_ortho_matrix,
            torch.eye(mrank).to(cavs_ortho_matrix.device),
            atol=1e-3,
            rtol=1e-3,
        ), f"Q^TQ is not identity matrix! Please check the CAVs matrix. {cavs_ortho_matrix.T @ cavs_ortho_matrix}"
        y_x_avs = y_all @ cavs_ortho_matrix @ cavs_ortho_matrix.T
        y_remainder = y_all - y_x_avs  # [# batches, # dims]

        dim_projected = y_projected.shape[1] if y_projected is not None else 0

        if mute_x_avs:
            y_x_avs.register_hook(lambda grad: torch.zeros_like(grad))
        if mute_remainder:
            y_remainder.register_hook(lambda grad: torch.zeros_like(grad))

        return (
            y_x_avs[:, :dim_projected] + y_remainder[:, :dim_projected],
            y_x_avs[:, dim_projected:] + y_remainder[:, dim_projected:],
        )
