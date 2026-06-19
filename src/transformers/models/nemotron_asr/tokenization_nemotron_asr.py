# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..parakeet.tokenization_parakeet import ParakeetTokenizer


class NemotronAsrTokenizer(ParakeetTokenizer):
    """
    BPE tokenizer for Nemotron 3.5 ASR. Inherits from [`ParakeetTokenizer`]. Because this is an RNN-T model (not
    CTC), consecutive-token grouping is disabled by default in `_decode`; the RNN-T decoder already emits one token
    per (non-blank) step, and the blank token doubles as the pad token and is filtered out.
    """

    def _decode(
        self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None, group_tokens=False, **kwargs
    ):
        return super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            group_tokens=group_tokens,
            **kwargs,
        )


__all__ = ["NemotronAsrTokenizer"]
