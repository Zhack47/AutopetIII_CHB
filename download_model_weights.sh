curl https://github.com/Zhack47/AutopetIII_CHB/releases/download/v0.1/FDG_weights.zip
curl https://github.com/Zhack47/AutopetIII_CHB/releases/download/v0.1/PSMA_weights.zip
unzip FDG_weights
unzip PSMA_weights
cp Dataset514_AUTOPETIII_SW_PSMA nnNUNet_results/
cp Dataset513_AUTOPETIII_SW_FDG nnUNet_results/
