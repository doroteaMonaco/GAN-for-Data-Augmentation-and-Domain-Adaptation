# STRUCTURE
This folder contains the following:
- raw: the original images from ISIC dataset
- processed: 
    - baseline: the initial 10k vs 1k split with train/test/val
    - augmented: baseline + synthetic (malign) samples
    - domain_adaptation: imbalanced classification with class-specific domain shift
- synthetic: 
    - gan_v*: malignant samples generated in a specific GAN version