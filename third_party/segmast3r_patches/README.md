# segmast3r patches

These two modules are ours, but at runtime they must live **inside** the
`third_party/segmast3r` checkout, under `src/models/mast3r_segfeat/`, because
that is where the SegMASt3R package imports them from:

```python
from src.models.mast3r_segfeat.segment_attention import SegmentAttentionLayer
from src.models.mast3r_segfeat.double_softmax_matcher import DoubleSoftmaxMatcher
```

Inside that checkout they are untracked, so a fresh clone or a re-checkout of
the dependency loses them silently and `MODEL.ARCH = lightglue` /
`lightglue_v2` then fails at import. This directory is the authoritative copy;
`setup_third_party.sh` deploys it.

| file | used by |
|---|---|
| `segment_attention.py` | SegMASt3R segment cross-attention (LG v1 head) |
| `double_softmax_matcher.py` | all SegMASt3R and SegVGGT LightGlue-style heads |

Re-run `bash setup_third_party.sh` after any update of the segmast3r checkout.
