# AlexNet-again

Aims of this project:

• Re-implemented AlexNet, a landmark deep CNN introduced in 2012, and trained it on the ILSVRC 2012 (ImageNet) dataset.
• Applied modern training techniques including cosine-decay learning-rate scheduling, He initialization, mixed-precision training, and data
augmentation, achieving 2 percentage points lower Top-1 error than the original AlexNet.
• Distributed AlexNet across 2 GPUs via layer-wise model parallelism and optimized inter-GPU data flow, achieving >70% faster training
compared with single-GPU execution.
• Extended training to higher epoch counts and tuned the optimization pipeline to improve model convergence and final validation accuracy.