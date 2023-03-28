#Adapted from LABMLAI

from typing import List

import torch

class MLM:
    """
    ## Masked LM (MLM)
    This class implements the masking procedure for a given batch of token sequences.
    """

    def __init__(self, *,
                 padding_token: int, mask_token: int, no_mask_tokens: List[int], n_tokens: int,
                 masking_prob: float = 0.15, randomize_prob: float = 0.1, no_change_prob: float = 0.1,
                 ):
        """
        * `padding_token` is the padding token `[PAD]`.
          We will use this to mark the labels that shouldn't be used for loss calculation.
        * `mask_token` is the masking token `[MASK]`.
        * `no_mask_tokens` is a list of tokens that should not be masked.
        This is useful if we are training the MLM with another task like classification at the same time,
        and we have tokens such as `[CLS]` that shouldn't be masked.
        * `n_tokens` total number of tokens (used for generating random tokens)
        * `masking_prob` is the masking probability
        * `randomize_prob` is the probability of replacing with a random token
        * `no_change_prob` is the probability of replacing with original token
        """
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[seq_len, batch_size]`.
        """

        # Mask `masking_prob` of tokens
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob
        # Unmask `no_mask_tokens`
        for t in self.no_mask_tokens:
            full_mask &= x != t

        # A mask for tokens to be replaced with original tokens
        unchanged = full_mask & (torch.rand(x.shape, device=x.device) < self.no_change_prob)
        # A mask for tokens to be replaced with a random token
        random_token_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        # Indexes of tokens to be replaced with random tokens
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        # Random tokens for each of the locations
        random_tokens = torch.randint(0, self.n_tokens, (len(random_token_idx[0]),), device=x.device)
        # The final set of tokens that are going to be replaced by `[MASK]`
        mask = full_mask & ~random_token_mask & ~unchanged

        # Make a clone of the input for the labels
        y = x.clone()

        # Replace with `[MASK]` tokens;
        # note that this doesn't include the tokens that will have the original token unchanged and
        # those that get replace with a random token.
        x.masked_fill_(mask, self.mask_token)
        # Assign random tokens
        x[random_token_idx] = random_tokens

        # Assign token `[PAD]` to all the other locations in the labels.
        # The labels equal to `[PAD]` will not be used in the loss.
        y.masked_fill_(~full_mask, self.padding_token)

        # Return the masked input and the labels
        return x, y