# ECNet project
we propose the elemental convolution (EC) operation to obtain a more general and global element-wise representations, and develop EC graph neural networks (ECNet) to accurately model material properties. We demonstrate that the ECNet models show better prediction in properties like band gaps, refractive index, and elastic moduli of crystals. The element-wise feature vectors are able to capture information of elemental types and crystal structures and show a special characteristics related to its target properties.

| performance | units    | ECSTL |  ECMTL | MODNet | MEGNet| training data|
|-------------|----------|--------|-------|-------|-------|--------------|
|Ef           | eV/atom  | 0.076  | 0.084  | 0.044 |  0.028 | 60000      |
|Eg           |eV        | 0.164  | 0.227   | 0.34 | 0.38  |   60000      |
|Eg^{nz}          |eV        | 0.27  | 0.27 |0.45  | 0.38  |   37179      |
| K_VRH       |log10(GPa)| 0.05  | 0.05    | - | 0.05  |   4722|
| G_VRH       |log10(GPa)| 0.049  | 0.046| -      | 0.079 |4722|
|n           |         | 0.046|-  | 0.0.05   | 0.08  |   3272      |



# Applications in HEAs
We use the ECNet to model the properties in the high entropy alloy systems.
ECNet could reach the state-of-the-art performance in predicting different properties and The model uses the hierarchy of the alloy systems, which utilize the low-component information to predict high-component properties.

# Environment Requirements
The basic environment is the PyTorch, and CUDA for your GPU machines. The projects use the graph to model pairwise relations between nodes, and the graph representation is then combined with the message passing networks. Thus, the framework is also based on the torch_geometric, where its installation could be refered to the official website [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). For other packages, you can easily install them via :

```bash
pip install -r package.txt 
```

# Citing the ECNet

If you use the codes in your work, please consider citing:

```bash

@article{wang_element-wise_2022,
  author = {Wang, Xinming and Tran, Nguyen-Dung and Zeng, Shuming and Hou, Cong and Chen, Ying and Ni, Jun},
	title = {Element-wise representations with {ECNet} for material property prediction and applications in high-entropy alloys},
  journal = {npj Computational Materials},
	volume = {8},
	issn = {2057-3960},
	doi = {10.1038/s41524-022-00945-x},
	number = {1},
	month = dec,
	year = {2022},
	pages = {253},
}


```
