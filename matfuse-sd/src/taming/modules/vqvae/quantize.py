import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer2(nn.Module):
    """Subset of taming-transformers VectorQuantizer2 used by MatFuse inference."""

    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.remap = remap
        self.unknown_index = unknown_index
        self.sane_index_shape = sane_index_shape
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        if rescale_logits or return_logits:
            raise ValueError("This VectorQuantizer2 compatibility layer does not return logits")

        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, self.embedding.weight.t())
        )
        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if self.legacy:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.view(
                z_q.shape[0], z_q.shape[2], z_q.shape[3]
            )

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embedding.weight)
