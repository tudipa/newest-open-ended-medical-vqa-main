# Training Log From Checkpoints

Generated: 2026-04-25 17:21:09 +10:00
Total runs: 15

| job_id | job_name | setting | model | dataset | epochs | lr | dropout | checkpoint_best | checkpoint_latest |
|---:|:---|:---|:---|:---|---:|:---|:---|:---|:---|
| 8080 | slake_train | frozen | gpt2-xl | slake | 30 | 5e-3 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8080/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8080/open_ended_latest.pt |
| 8122 | slake_train_1e4 | frozen | gpt2-xl | slake | 30 | 1e-4 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8122/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8122/open_ended_latest.pt |
| 8161 | slake_train_1e4_50_epochs | frozen | gpt2-xl | slake | 50 | 1e-4 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8161/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8161/open_ended_latest.pt |
| 8162 | slake_train_5e3_50_epochs | frozen | gpt2-xl | slake | 50 | 5e-3 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8162/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8162/open_ended_latest.pt |
| 8257 | slake_train_5e3_30ep_first_commit_appr | frozen | gpt2-xl | slake | 30 | 5e-3 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8257/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8257/open_ended_latest.pt |
| 8329 | slake_train_1e4_30ep_1st_comm_appr | frozen | gpt2-xl | slake | 30 | 1e-4 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8329/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8329/open_ended_latest.pt |
| 8342 | slake_train_5e3_100ep_first_commit_appr | frozen | gpt2-xl | slake | 100 | 5e-3 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8342/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_8342/open_ended_latest.pt |
| 9078 | slake_train_5e3_100ep_after_update_to_improve_bert | frozen | gpt2-xl | slake | 100 | 5e-3 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9078/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9078/open_ended_latest.pt |
| 9120 | slake_train_5e3_200ep_after_update_to_improve_bert | frozen | gpt2-xl | slake | 200 | 5e-3 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9120/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9120/open_ended_latest.pt |
| 9127 | slake_train_1e4_200ep_after_update_to_improve_bert | frozen | gpt2-xl | slake | 200 | 1e-4 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9127/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9127/open_ended_latest.pt |
| 9153 | slake_train_1e4_200ep_after_update_dropout | frozen | gpt2-xl | slake | 200 | 1e-4 |  | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9153/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9153/open_ended_latest.pt |
| 9173 | slake_train_5e3_30ep_after_update_dropout_to_cli_to_compare_with_8080_checkpoint | frozen | gpt2-xl | slake | 30 | 5e-3 | 0.5 | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9173/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9173/open_ended_latest.pt |
| 9174 | slake_train_lora_5e3_30ep | lora | gpt2-xl | slake | 30 | 5e-3 | 0.5 | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9174/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9174/open_ended_latest.pt |
| 9180 | slake_train_5e3_100ep_after_update_dropout_to_cli_to_compare_with_8080_checkpoint | frozen | gpt2-xl | slake | 100 | 5e-3 | 0.5 | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9180/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9180/open_ended_latest.pt |
| 9214 | slake_train_lora_1e4_100ep | lora | gpt2-xl | slake | 100 | 1e-4 | 0.5 | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9214/open_ended_best_val.pt | /home/s225507154/checkpoints/newest-open-ended-medical-vqa-main/job_9214/open_ended_latest.pt |

## Notes
- Source files: `checkpoints/run_config_*.txt` and local `logs/slurm/slake_train_*.out` when run_config is unavailable.
- Full parameter string is in the CSV column `train_args`.
