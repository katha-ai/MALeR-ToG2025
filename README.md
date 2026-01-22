# MALeR: Improving Compositional Fidelity in Layout Guided Generation

Official codebase built over Bounded-Attention. For tracking MALeR losses addition to the Bounded Attention's bounded guidance loss. Refer to https://github.com/katha-ai/MALeR-ToG2025/pull/1, [file changes](https://github.com/katha-ai/MALeR-ToG2025/pull/1/files)

> Recent advances in text-to-image models have enabled a new era of creative and controllable image generation. However, generating compositional scenes with multiple subjects and attributes remains a significant challenge. To enhance user control over subject placement, several layout-guided methods have been proposed. However, these methods face numerous challenges, particularly in compositional scenes. Unintended subjects often appear outside the layouts, generated images can be out-of-distribution and contain unnatural artifacts, or attributes bleed across subjects, leading to incorrect visual outputs. In this work, we propose MALeR, a method that addresses each of these challenges. Given a text prompt and corresponding layouts, our method prevents subjects from appearing outside the given layouts while being in-distribution. Additionally, we propose a masked, attribute-aware binding mechanism that prevents attribute leakage, enabling accurate rendering of subjects with multiple attributes, even in complex compositional scenes. Qualitative and quantitative evaluation demonstrates that our method achieves superior performance in compositional accuracy, generation consistency, and attribute binding compared to previous work. MALeR is particularly adept at generating images of scenes with multiple subjects and multiple attributes per subject.

<a href="https://arxiv.org/abs/2511.06002"><img src="https://img.shields.io/badge/arXiv-2511.06002-b31b1b.svg"></a>
<a href="https://katha-ai.github.io/projects/maler/"><img src="https://img.shields.io/badge/Project-Website-orange"></a>

<p align="center">
<img src="images/MALeR_FastForwardVideo.gif" width="800px"/>
</p>


### Setting-up the environment
Create an environment of your choice, simply run
```
conda create --name maler python=3.11.4
conda activate maler
pip install -r requirements.txt
```

### Usage
To generate images, simple can use `run_maler.py` and call `run` function to generate images

```
boxes = [
    [0.14, 0.67, 0.34, 0.91],
    [0.60, 0.55, 0.85, 0.91],
]

prompt = "A professional studio photograph of a red crystal bear on the left and a blue marble rabbit on the right. 8k, white background"
subject_token_indices = [[7,8,9],[15,16,17]]

run(
    boxes,
    prompt,
    subject_token_indices,
    out_dir=f"./outputs/test",
    seed=121,
    init_step_size=30,
    final_step_size=8,
    num_guidance_steps=15,
    lambda_reg=0.01,
    lambda_kl=5,
    early_iterations=5,
    early_gd_iterations=5,
    reg_type=False,
    sym_kl=1,
    dissim=1
)
```
The `run` function receives the following extra parameters:
- sym_kl: scale for controlling subject attribute similarity
- dissim: scale for controlling subject attribute dissimilarity
- lambda_reg: masked regularization scale
- lambda_kl: kl alignment scale

### Acknowledgements 
The code was built on top of the code from the following repository:
- [bounded-attention](https://github.com/omer11a/bounded-attention)

### Cite
If you find this repository useful, please cite the following paper

```
@article{saxena2025maler,
  title={MALeR: Improving Compositional Fidelity in Layout-Guided Generation},
  author={Saxena, Shivank and Srivastava, Dhruv and Tapaswi, Makarand},
  journal={ACM Transactions on Graphics (TOG)},
  volume={44},
  number={6},
  pages={1--12},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```
