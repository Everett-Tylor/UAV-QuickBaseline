# Round17: research-informed barren contrast experiment

## Primary papers and applicability

- Wang et al., Exploring Cross-Image Pixel Contrast for Semantic Segmentation,
  ICCV2021: https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.html
  Learns similar same-class and separated different-class pixel embeddings.
  General segmentation evidence, not a guaranteed barren gain on our competition.
- Liu et al., Bootstrapping Semantic Segmentation with Regional Contrast,
  ICLR2022 (ReCo): https://arxiv.org/abs/2104.04465
  Sparse difficult examples for regional contrast; memory-efficient inspiration.
- Wang et al., LoveDA, NeurIPS2021 datasets/benchmarks:
  https://arxiv.org/abs/2110.08733
  Contains barren, background, forest and agriculture. Highlights complex
  background and domain/class-distribution differences. This is diagnostic
  context, not evidence of a particular guaranteed improvement here.

No external dataset is downloaded or used. These papers do not establish an
effective barren-specific gain on our data; this experiment tests that hypothesis.

## Implementation and controlled comparison

Lightweight prototype contrast on existing160-channel joint decoder features.
Not a full reproduction of ReCo or ContrastiveSeg: no projection head or memory
bank. Classes1/5/6/7; feature-cell label purity>=.9; ignore0 excluded; >=8cells
per class. Only apply when barren and another class both present. Detached
normalized class-mean prototypes, max64 most uncertain queries/class, cosine
softmax temperature.2. Auxiliary coefficient.03 ramps over first epoch.
No new inference parameters, no ensemble.

Both contrast and zero-weight continuation control start round15 best (official
68.75), never round16 which trained on old validation. Same seed20261008,
6296/700 split,20% existing hard crops,3epochs640,batch4accum2,lr5e-7/head5e-6.
Different loss is the intended experimental change. Finite loss/gradients,
ignored-cell gradient, absent-class and separated-feature tests pass.

Both evaluated with identical4scale+flip+brightness settings. Contrast export
requires no overall drop and higher barren IoU than both control and round15,
and dark/contrast groups no more than.2pp below round15. Full confusion matrices
are retained. Local improvement is not official score evidence; repeated use of
the same holdout can still cause selection bias.

Run: python outputs/UAV-QuickBaseline/round17_contrast_pipeline.py
Status/logs: outputs/round17_contrast. Two independent3epoch training runs,
executed sequentially; only one checkpoint can be exported.
