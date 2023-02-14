rm -rf ./sd_v2-1_truss/data
git lfs install --skip-smudge 
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base ./sd_v2-1_truss/data
# cd ./sd_v2-1_truss/data; git lfs pull --include v2-1_512-nonema-pruned.ckpt