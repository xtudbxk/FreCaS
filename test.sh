# sd21 -- 1024
python3 main.py --gs 7.5 --prompts prompts.lst --tsize [[512,512],[1024,1024]] --msp_endtimes 100 0 --msp_steps 40 10 --msp_gamma 3.0 --name sd21 --images-per-prompt 1 --facfg_weight 45.0 7.5 --camap_weight 0.6 --output results_sd21_1024
# sd21 - 2048
python3 main.py --gs 7.5 --prompts prompts.lst --tsize [[512,512],[1024,1024],[2048,2048]] --msp_endtimes 200 200 0 --msp_steps 30 10 10 --msp_gamma 3.0 --name sd21 --images-per-prompt 1 --facfg_weight 35.0 7.5 --camap_weight 0.4 --output results_sd21_2048

# sdxl -- 2048
python3 main.py --gs 7.5 --prompts prompts.lst --tsize [[1024,1024],[2048,2048]] --msp_endtimes 200 0 --msp_steps 40 10 --msp_gamma 1.5 --name sdxl --images-per-prompt 1 --facfg_weight 35.0 7.5 --camap_weight 0.6 --output results_sdxl_2048
# sdxl -- 4096
python3 main.py --gs 7.5 --prompts prompts.lst --tsize [[1024,1024],[2048,2048],[4096,4096]] --msp_endtimes 400 200 0 --msp_steps 30 5 15 --msp_gamma 2.0 --name sdxl --images-per-prompt 1 --facfg_weight 35.0 7.5 --camap_weight 0.6 --output results_sdxl_4096 --vae_tiling

# sd3 -- 2048
python3 main.py --gs 7.5 --prompts prompts.lst --tsize [[1024,1024],[2048,2048]] --msp_endtimes 50 0 --msp_steps 20 8 --name sdxl --images-per-prompt 1 --facfg_weight 35.0 7.5 --camap_weight 0.5 --output results_sd3_2048
