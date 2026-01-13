from run_maler import run

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
