# FID results

| run | mode | predictor | ckpt_dir | last_log_step | traj_n | ckpt_bytes | FID |
|---|---|---|---|---|---|---|---|
| linear_noise | linear | noise | results/predictor_noise/beta_linear/09-22-151536 | 98000 | 50 | 235428141 | **15.321049025808424** |
| quad_noise | quad | noise | results/predictor_noise/beta_quad/09-22-155004 | 98000 | 50 | 235428141 | **6.750266360192359** |
| cosine_noise | cosine | noise | results/predictor_noise/beta_cosine/09-23-063925 | 98000 | 50 | 235428141 | **7.767465895530291** |
| linear_x0 | linear | x0 | results/predictor_x0/beta_linear/09-23-153116 | 98000 | 50 | 235428141 | **31.673476735828146** |
| linear_mean | linear | mean | results/predictor_mean/beta_linear/09-23-232559 | 98000 | 50 | 235428141 | **95.49943835493106** |

註：每個 run 皆為 100,000 steps、batch 16、log_interval 2000。
last_log_step 應為 98000、traj_n 應為 50、ckpt_bytes 應為 235428141，
三者皆符合才代表該 run 完整跑完。

linear_noise 另有一次獨立抽樣得 FID 15.706（同一 checkpoint），
用以估計抽樣變異約 0.4。

產生時間：2026-09-24 08:50:19
