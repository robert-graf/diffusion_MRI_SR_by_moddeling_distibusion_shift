# Modeling the acquisition shift between axial and sagittal MRI for diffusion superresolution to enable axial spine segmentation

This is the official code for "Modeling the acquisition shift between axial and sagittal MRI for diffusion superresolution to enable axial spine segmentation"

### Installation
```bash
conda create --name mri_sr python=3.10
conda activate mri_sr
pip install -r requirements.txt
# Go to https://pytorch.org/get-started/locally/ and install pytorch for your GPU version first. If this version is not working.
```
You may have to install "pip install protobuf==3.20"


### Make a dataset

Follow the instruction in "make_dataset.ipynb"

### Run Training
You can start the script with commandant arguments and/or a conf file.

```python train_DAE.py --config /config/my_dataset.conf```


```sage: train_DAE.py [-h] [-config CONFIG] [--RandomInterlaceMovementArtifactFactor RANDOMINTERLACEMOVEMENTARTIFACTFACTOR] [--ablation_lvl ABLATION_LVL] [--add_palette_condition_to_encoding]
                    [--attention_resolutions ATTENTION_RESOLUTIONS [ATTENTION_RESOLUTIONS ...]] [--batch_size BATCH_SIZE] [--batch_size_eval BATCH_SIZE_EVAL] [--beta1 BETA1] [--beta2 BETA2]
                    [--beta_schedule {linear,cosine}] [--dataset DATASET] [--dataset_val DATASET_VAL] [--debug] [--dims DIMS] [--discriminator DISCRIMINATOR] [--dropout DROPOUT] [--ds_type DS_TYPE]
                    [--early_stopping_patience EARLY_STOPPING_PATIENCE] [--ema_decay EMA_DECAY] [--embed_channels EMBED_CHANNELS] [--enc_channel_mult ENC_CHANNEL_MULT [ENC_CHANNEL_MULT ...]]
                    [--enc_num_res_blocks ENC_NUM_RES_BLOCKS] [--experiment_name EXPERIMENT_NAME] [--fast_dev_run] [--fp32] [--generative_type {ddpm,ddim}] [--gpus GPUS [GPUS ...]] [--grad_clip GRAD_CLIP]
                    [--hessian_penalty HESSIAN_PENALTY] [--image_name IMAGE_NAME] [--img_size IMG_SIZE [IMG_SIZE ...]] [--in_channels IN_CHANNELS] [--inpainting {random_ege,perlin}] [--linspace] [--log_dir LOG_DIR]
                    [--loss_type {mse,l1}] [--lr LR] [--model_channels MODEL_CHANNELS] [--model_mean_type {eps}] [--model_name {beatgans_ddpm,autoencoder}]
                    [--model_type {ddpm,autoencoder,palette_only,RRDBNet,RRDBNet_diffusion,RCAN,Pix2Pix}] [--model_var_type {fixed_small,fixed_large}] [--monitor MONITOR]
                    [--net_ch_mult NET_CH_MULT [NET_CH_MULT ...]] [--net_enc_pool {adaptivenonzero,ConvEmb}] [--new] [--num_cpu NUM_CPU] [--num_timesteps NUM_TIMESTEPS] [--num_timesteps_ddim NUM_TIMESTEPS_DDIM]
                    [--optimizer {adam,adamW}] [--overfit] [--palette_condition PALETTE_CONDITION [PALETTE_CONDITION ...]] [--prob_transforms PROB_TRANSFORMS] [--rescale_timesteps]
                    [--save_every_samples SAVE_EVERY_SAMPLES] [--schedular SCHEDULAR] [--seed SEED] [--sr_source SR_SOURCE [SR_SOURCE ...]] [--sr_target SR_TARGET] [--target_batch_factor TARGET_BATCH_FACTOR]
                    [--total_samples TOTAL_SAMPLES] [--train_mode {diffusion,ESRGANModel,RCAN,Pix2Pix}]
                    [--transforms {CenterCrop,CenterCrop256,RandomHorizontalFlip,RandomVerticalFlip,pad,random_crop,resize,to_RGB} [{CenterCrop,CenterCrop256,RandomHorizontalFlip,RandomVerticalFlip,pad,random_crop,resize,to_RGB} ...]]
                    [--transforms_3D {ColorJitter3D,Fork_smore_many_scale,Fork_smore_many_scale_stop_gap,Fork_smore_one_scale,Fork_upscale_only,RandomBiasField,RandomBlur,RandomExponentialHistogramTransform,RandomExponentialHistogramTransform_only_lr,RandomInterlaceMovementArtifact,RandomNoise,RandomNoiseMedium,RandomNoiseStrong,RandomQuadraticHistogramTransform,RandomQuadraticHistogramTransform_only_lr,RandomRotate,RandomScale,_Noise,_RandomBlur} [{ColorJitter3D,Fork_smore_many_scale,Fork_smore_many_scale_stop_gap,Fork_smore_one_scale,Fork_upscale_only,RandomBiasField,RandomBlur,RandomExponentialHistogramTransform,RandomExponentialHistogramTransform_only_lr,RandomInterlaceMovementArtifact,RandomNoise,RandomNoiseMedium,RandomNoiseStrong,RandomQuadraticHistogramTransform,RandomQuadraticHistogramTransform_only_lr,RandomRotate,RandomScale,_Noise,_RandomBlur} ...]]
                    [--val_check_interval VAL_CHECK_INTERVAL] [--val_niis VAL_NIIS [VAL_NIIS ...]] [--val_niis_every_epoch VAL_NIIS_EVERY_EPOCH] [--val_niis_every_epoch_start VAL_NIIS_EVERY_EPOCH_START]
                    [--val_niis_pairs VAL_NIIS_PAIRS [VAL_NIIS_PAIRS ...]] [--weight_decay WEIGHT_DECAY] [--x_start X_START]

```


|key       | description                                 |   type   |
|----------|---------------------------------------------|----------|
|-h, --help| show this help message and exit||
|-config CONFIG, --config| config file path| path|
|--experiment_name| EXPERIMENT_NAME| str|
|--new| prevents reloading if you continue training on the same name|
|-lr | Learning Rate|              float
|--batch_size |BATCH_SIZE|int
|--batch_size_eval| BATCH_SIZE_EVAL |int
|--target_batch_factor| how many batches should gradient accumulation uses|int|
|--total_samples| after how many samples stop the training|int
|--gpus| id or list of ids of the gpu to use| int/list of int
|--num_cpu| number of cpus for loading images| int|
|--log_dir| name of the logdir, where the tensorboard and weights are stored| str|
|weight_decay| weight_decay| float|
|dataset| path to the dataset or keyword of the dataset|str
|dataset_val|path to the dataset or keyword of the dataset|str
|ds_type| This key changes the dataloader to support other files than jpg| str "csv_2D_super"|
|--dims | DIMS 2 for 2D or 3 for 3D|           int
|--optimizer| change optimizer| {adam,adamW}|
|  --beta1 |BETA1 of optimizer|         float
|  --beta2 |BETA2 of optimizer|         float
|x_start| key the dataloader returns for output of the DDIM| "hq"
|image_name| key the dataloader returns for input of the DAE| : "img_lr"
|palette_condition|List of image keys that are concatenated to the noised image |["img_lr"]
|--beta_schedule | DDPM schedular |{linear,cosine}|
|sr_target| The initial resolution in mm of the image| float
|sr_source| The resolution the Up/Down direction of the jpg is reduced to| list[float]
|train_mode|keys to change the training|"diffusion","ESRGANModel","RCAN"|
|model_type||"palette_only"|
|model_name|change the model backbone| "ddpm","autoencoder","palette_only","RRDBNet","RRDBNet_diffusion","RCAN"
|ablation_lvl| gives a fixed preprocessing depending on the number. See also --transforms_3D| default = 11|
|attention_resolutions| on whitch resolution should be a ConvAttention| list of int|
| --fp32| uses 16 bit by default, Is not supported for everything. use this flag to go back to 32 bit| 
|--grad_clip |
|--img_size| Crop size of the image| int
|--in_channels| Number of channels of the image (usually one for MRI)| int
|--model_channels| base channel number|int
|--net_ch_mult| channel multiplier for every layer |List of int
|--prob_transforms| base probability a argumentation is used| float
|--RandomInterlaceMovementArtifactFactor| change the base probability the rima is used| float
-------------------------------------------------------------------------------
Others, May be used by some models
```
--add_palette_condition_to_encoding
--discriminator
--dropout 
--ema_decay
--embed_channels
--enc_channel_mult
--enc_num_res_blocks
--fast_dev_run
--hessian_penalty
--inpainting {random_ege,perlin}
--linspace #For 3D stuff
--loss_type {mse,l1}
--model_mean_type {eps}
--model_var_type {fixed_small,fixed_large}
--monitor str
--net_enc_pool {adaptivenonzero,ConvEmb}
--num_timesteps NUM_TIMESTEPS
--num_timesteps_ddim NUM_TIMESTEPS_DDIM
--rescale_timesteps
--save_every_samples
--seed
--transforms_3D  list of {ColorJitter3D,Fork_smore_many_scale,Fork_smore_many_scale_stop_gap,Fork_smore_one_scale,Fork_upscale_only,RandomBiasField,RandomBlur,RandomExponentialHistogramTransform,RandomExponentialHistogramTransform_only_lr,RandomInterlaceMovementArtifact,RandomNoise,RandomNoiseMedium,RandomNoiseStrong,RandomQuadraticHistogramTransform,RandomQuadraticHistogramTransform_only_lr,RandomRotate,RandomScale,_Noise,_RandomBlur}
--val_check_interval
```

### Run Inference

See inference.ipynb