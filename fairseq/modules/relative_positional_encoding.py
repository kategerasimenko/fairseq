import torch


class RelEncoding(torch.nn.Module):
    def __init__(self, max_relative_position, num_units):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position  # The maximum relative position to represent
        self.weight = torch.nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inpt, max_seq_len, transpose_weight=True):
        relative_positions = self._relative_positions(max_seq_len)
        weight = self.weight[relative_positions]

        # inpt: Tensor with shape :math:`[batch * num_heads, max_seq_len, self.num_units]`.
        # weight: Tensor with shape :math:`[max_seq_len, max_seq_len, self.num_units]`.
        # out: Tensor with shape: math: `[batch * num_heads, max_seq_len, max_seq_len]`.

        inpt = inpt.transpose(0, 1)
        if transpose_weight:
            weight = torch.permute(weight, dims=[0, 2, 1])

        out = torch.matmul(inpt, weight)
        out = out.transpose(0, 1)
        return out

    def _relative_positions(self, max_seq_len):
        """Builds the relative positions.
        Args:
            max_seq_len: The maximum length of the sequence.
        Returns:
            Positive relative positions with shape :math:`[max_seq_len, max_seq_len]`.
        """
        arange = torch.arange(max_seq_len)
        distance = torch.unsqueeze(arange, 0) - torch.unsqueeze(arange, 1)  # Distance to the diagonal.
        distance = torch.clamp(distance, -self.maximum_position, self.maximum_position)
        return distance + self.maximum_position  # Return positive indices.
