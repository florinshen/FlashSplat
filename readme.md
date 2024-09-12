
## FlashSplat


### [Arxiv](https://arxiv.org) | [Demo]()


> This study addresses the challenge of accurately segmenting 3D Gaussian Splatting from 2D masks. Conventional methods often rely on iterative gradient descent to assign each Gaussian a unique label, leading to lengthy optimization and sub-optimal solutions. Instead, we propose a straightforward yet globally optimal solver for 3D-GS segmentation. The core insight of our method is that, with a reconstructed 3D-GS scene , the rendering of the 2D masks is essentially a linear function with respect to the labels of each Gaussian. As such, the optimal label assignment can be solved via linear programming in closed form. This solution capitalizes on the alpha blending characteristic of the splatting process for single step optimization. By incorporating the background bias in our objective function, our method shows superior robustness in 3D segmentation against noises. Remarkably, our optimization completes within 30 seconds, about 50x faster than the best existing methods.



### Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [gaussian-grouping](https://github.com/lkeab/gaussian-grouping)
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [diff-gaussian-rasterization-kiui](https://github.com/ashawkey/diff-gaussian-rasterization.git)

### Citation

```bibtex
@article{flashsplat,
  title={FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally},
  author={Shen, Qiuhong and Yang, Xingyi and Wang, Xinchao},
  journal={European Conference of Computer Vision},
  year={2024}
}
```

