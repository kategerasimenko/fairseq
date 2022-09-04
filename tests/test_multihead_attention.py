# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.relative_positional_encoding import RelEncoding


class TestMultiheadAttention(unittest.TestCase):
    def test_append_prev_key_padding_mask(self):
        bsz = 1
        src_len = 4

        cases = [
            # no padding mask
            (None, None, None),
            # current padding mask only
            (
                torch.tensor([[1]]).bool(),
                None,
                torch.tensor([[0, 0, 0, 1]]).bool(),
            ),
            # previous padding mask only
            (
                None,
                torch.tensor([[0, 1, 0]]).bool(),
                torch.tensor([[0, 1, 0, 0]]).bool(),
            ),
            # both padding masks
            (
                torch.tensor([[1]]).bool(),
                torch.tensor([[0, 1, 0]]).bool(),
                torch.tensor([[0, 1, 0, 1]]).bool(),
            ),
        ]
        for c in cases:
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                c[0],
                c[1],
                batch_size=bsz,
                src_len=src_len,
                static_kv=False,
            )

            if key_padding_mask is not None:
                self.assertTrue(
                    torch.all(torch.eq(key_padding_mask, c[2])),
                    f'Unexpected resultant key padding mask: {key_padding_mask}'
                    f' given current: {c[0]} and previous: {c[1]}',
                )
                self.assertEqual(key_padding_mask.size(0), bsz)
                self.assertEqual(key_padding_mask.size(1), src_len)
            else:
                self.assertIsNone(c[2])

    def test_relative_positions(self):
        positions = RelEncoding(max_relative_position=2, num_units=2)._relative_positions(4)
        self.assertEqual(positions, [[2, 3, 4, 4], [1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 1, 2]])

    def test_attention_with_relative_positions(self):
        def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
            if maxlen is None:
                maxlen = lengths.max()
            row_vector = torch.arange(0, maxlen, 1)
            matrix = torch.unsqueeze(lengths, dim=-1)
            mask = row_vector < matrix
            mask.type(dtype)
            return mask

        attention = MultiheadAttention(4, 20, maximum_relative_position=6)
        x = torch.rand([2, 9, 10])
        mask = sequence_mask([9, 7])
        y = attention(x, mask=mask)

    def test_attention_with_relative_positions_cache(self):
        attention = MultiheadAttention(4, 20, maximum_relative_position=6)
        x = torch.rand([4, 1, 10])
        cache = (torch.zeros([4, 4, 0, 5]), torch.zeros([4, 4, 0, 5]))
        _, cache = attention(x, cache=cache)


if __name__ == '__main__':
    unittest.main()
