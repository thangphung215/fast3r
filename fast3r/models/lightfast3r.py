import cv2
import torchvision
import time
import timm
import torch
import numpy as np
import torch.nn as nn
import huggingface_hub
import torch.distributed
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from copy import deepcopy
from typing import Optional
from einops import rearrange
from packaging import version
from functools import partial
from fast3r.utils import pylogger
from transformers import AutoModel
from omegaconf import DictConfig, OmegaConf
from efficientnet_pytorch import EfficientNet
from fast3r.dust3r.patch_embed import get_patch_embed
from fast3r.dust3r.heads.postprocess import postprocess
from fast3r.dust3r.heads.dpt_head import PixelwiseTaskWithDPT
from fast3r.croco.models.blocks import Block, PositionGetter
from fast3r.dust3r.datasets.base.base_stereo_view_dataset import view_name
from fast3r.croco.models.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from fast3r.models.components.llama import TransformerBlock, RMSNorm, precompute_freqs_cis

from fast3r.models.fast3r import CroCoEncoder
from fast3r.dust3r.utils.misc import (
    freeze_all_params,
    transpose_to_landscape,
)

from fast3r.dust3r.datasets.base.base_stereo_view_dataset import view_name
from fast3r.dust3r.heads.postprocess import postprocess
from fast3r.dust3r.heads.dpt_head import PixelwiseTaskWithDPT
from fast3r.croco.models.blocks import Block
from fast3r.croco.models.pos_embed import (
    RoPE2D, get_1d_sincos_pos_embed_from_grid)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse(
    "0.22.0"
), "Outdated huggingface_hub version, please reinstall requirements.txt"



class LightFast3Rv2(nn.Module,
                    huggingface_hub.PyTorchModelHubMixin,
                    repo_url="",
                    tags=["image-to-3d"]
                    ):
    def __init__(
        self,
        encoder_args: dict,
        decoder_args: dict,
        head_args: dict,
        freeze="none",
    ):
        super(LightFast3Rv2, self).__init__()

        self.encoder_args = OmegaConf.to_container(encoder_args) if isinstance(
            encoder_args, DictConfig) else encoder_args
        self.build_encoder(encoder_args)

        self.decoder_args = OmegaConf.to_container(decoder_args) if isinstance(
            decoder_args, DictConfig) else decoder_args
        self.build_decoder(decoder_args)

        self.head_args = OmegaConf.to_container(head_args) if isinstance(
            head_args, DictConfig) else head_args
        self.build_head(head_args)

        # how many views to process in parallel in the head, used to avoid OOM
        self.max_parallel_views_for_head = 25

        self.set_freeze(freeze)

    def build_encoder(self, encoder_args: dict):
        # Initialize the encoder based on the encoder type
        if encoder_args["encoder_type"] == "darknet53":
            # Drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            # self.encoder = CroCoEncoder(**encoder_args)
            self.encoder = LightDarkNet53(**encoder_args)
        elif encoder_args["encoder_type"] == "resnet18":
            # Drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            self.encoder = ResNetEncoder(**encoder_args)
        elif encoder_args["encoder_type"] == "mambavision":
            # Drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            self.encoder = Mambavision(**encoder_args)
        elif encoder_args["encoder_type"] == 'efficientnet':  # Version 2
            encoder_args.pop('encoder_type')
            self.encoder = Efficient(**encoder_args)
        elif encoder_args["encoder_type"] == 'efficientnetv2':  # Version 2
            encoder_args.pop('encoder_type')
            self.encoder = EfficientV2(**encoder_args)
        elif encoder_args["encoder_type"] == 'mobilenetv4':  # Version 2
            encoder_args.pop('encoder_type')
            self.encoder = MobileNetV4(**encoder_args)
        elif encoder_args["encoder_type"] == 'mobilenetv4_167':  # Version 2
            encoder_args.pop('encoder_type')
            self.encoder = MobileNetV4_167(**encoder_args)
        elif encoder_args["encoder_type"] == 'efficientnetv2-l':
            encoder_args.pop('encoder_type')
            self.encoder = EfficientnetV2_L(**encoder_args)
        else:
            raise ValueError(
                f"Unsupported encoder type: {encoder_args['encoder_type']}")

    def build_decoder(self, decoder_args: dict):
        decoder_args["decoder_type"] = decoder_args.get(
            'decoder_type', 'fast3rcnn')  # default to fast3r if not specified
        if decoder_args["decoder_type"] == 'fast3rcnn':
            decoder_args = deepcopy(decoder_args)
            decoder_args.pop('decoder_type')
            self.decoder = Fast3RDecoderCNN(**decoder_args)
        elif decoder_args["decoder_type"] == 'fast3rcnn2':
            decoder_args = deepcopy(decoder_args)
            decoder_args.pop('decoder_type')
            self.decoder = Fast3RDecoderCNN2(**decoder_args)
        elif decoder_args["decoder_type"] == 'mamba':  # Version 3
            decoder_args.pop('decoder_type')
            self.decoder = MambaFusionDecoder(**decoder_args)
        else:
            raise ValueError(
                f"Unsupported decoder type: {decoder_args['decoder_type']}")

    def build_head(
        self,
        head_args: dict,
    ):
        self.output_mode = head_args['output_mode']  # pts3d
        self.head_type = head_args['head_type']  # dpt
        self.depth_mode = head_args['depth_mode']
        self.conf_mode = head_args['conf_mode']

        # allocate primary downstream head
        self.downstream_head = self.head_factory(
            head_args['head_type'], head_args['output_mode'],
            has_conf=bool(head_args['conf_mode']),
            patch_size=head_args['patch_size']  # 16
        )

        # add the second head if with_local_head is True
        if head_args.get('with_local_head', False):
            self.downstream_head_local = self.head_factory(
                head_args['head_type'], head_args['output_mode'],
                has_conf=bool(head_args['conf_mode']),
                patch_size=head_args['patch_size']
            )
        else:
            self.downstream_head_local = None

        # magic wrapper
        self.head = transpose_to_landscape(
            self.downstream_head, activate=head_args['landscape_only']
        )

        if self.downstream_head_local:
            self.local_head = transpose_to_landscape(
                self.downstream_head_local,
                activate=head_args['landscape_only']
            )
        else:
            self.local_head = None

    def head_factory(self, head_type, output_mode, has_conf=False,
                     patch_size=16):
        """ " build a prediction head for the decoder"""
        if head_type == "dpt" and output_mode == "pts3d":
            # assert self.decoder_args["depth"] > 9
            # l2 = self.decoder_args["depth"]
            if hasattr(self, 'decoder_args') and 'num_cnn_layers' in self.decoder_args:
                l2 = self.decoder_args["num_cnn_layers"]
            elif hasattr(self, 'decoder_args') and 'depth' in self.decoder_args:
                l2 = self.decoder_args["depth"]
            else:
                l2 = 12  # Default value
            assert l2 > 9
            feature_dim = 256
            last_dim = feature_dim // 2
            out_nchan = 3
            ed = self.encoder_args["embed_dim"]  # 1024
            dd = self.decoder_args["embed_dim"]  # 768
            return PixelwiseTaskWithDPT(
                num_channels=out_nchan + has_conf,
                feature_dim=feature_dim,
                last_dim=last_dim,
                hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                dim_tokens=[ed, dd, dd, dd],
                postprocess=postprocess,
                depth_mode=self.head_args["depth_mode"],
                conf_mode=self.head_args["conf_mode"],
                head_type="regression",
                patch_size=patch_size,
            )
        else:
            raise NotImplementedError(
                f"unexpected {head_type=} and {output_mode=}")

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)

    def load_from_dust3r_checkpoint(self, dust3r_checkpoint_path: str):
        """Load a Dust3R checkpoint into the model.
        Only load the patch_embed, enc_blocks, enc_norm, and downstream_head1 components from the checkpoint.

        Args:
            dust3r_checkpoint_path (str): Path to the Dust3R checkpoint.
        """
        # Load the checkpoint
        checkpoint = torch.load(dust3r_checkpoint_path,
                                weights_only=False)['model']

        # Initialize state dictionaries for different components
        encoder_state_dict = {}
        downstream_head_state_dict = {}

        # Prepare to track loaded keys
        loaded_keys = set()

        # Split the checkpoint into encoder and downstream head
        for key, value in checkpoint.items():
            if key.startswith("patch_embed") or key.startswith("enc_blocks") or key.startswith("enc_norm"):
                if isinstance(self.encoder, CroCoEncoder):
                    new_key = key.replace("patch_embed", "encoder.patch_embed") \
                                 .replace("enc_blocks", "encoder.enc_blocks") \
                                 .replace("enc_norm", "encoder.enc_norm")
                    encoder_state_dict[new_key] = value
                    loaded_keys.add(key)  # Tentatively mark as loaded
            elif key.startswith("downstream_head1"):
                new_key = key.replace("downstream_head1", "downstream_head")
                downstream_head_state_dict[new_key] = value
                loaded_keys.add(key)  # Tentatively mark as loaded

        # Load the encoder part into the model if it is an instance of CroCoEncoder
        if isinstance(self.encoder, CroCoEncoder):
            load_result = self.load_state_dict(
                encoder_state_dict, strict=False)

            # Remove keys that failed to load
            missing_keys = set(load_result.missing_keys)
            unexpected_keys = set(load_result.unexpected_keys)
            loaded_keys -= (missing_keys | unexpected_keys)

        # Load the downstream head part into the model with try-catch logic
        # Save the original downstream head state to restore in case of failure
        downstream_head_original_state = {
            k: v.clone() for k, v in self.downstream_head.state_dict().items()}

        if not self.head_args.get('skip_load_pretrained_head', False):
            try:
                load_result = self.load_state_dict(
                    downstream_head_state_dict, strict=False)

                # Remove keys that failed to load
                missing_keys = set(load_result.missing_keys)
                unexpected_keys = set(load_result.unexpected_keys)
                loaded_keys -= (missing_keys | unexpected_keys)
            except RuntimeError as e:
                log.warning(f"Error loading downstream head: {str(e)}")
                log.warning("Reverting downstream head to its original state")
                # Revert downstream head to its original state
                self.downstream_head.load_state_dict(
                    downstream_head_original_state)

                del downstream_head_original_state

                # Remove downstream head keys from loaded_keys, as they were not loaded
                loaded_keys -= set([key for key in checkpoint.keys()
                                   if key.startswith("downstream_head1")])
        else:
            log.info("Skipping loading pretrained head")

        # Compute not loaded keys as difference between all checkpoint keys and loaded keys
        checkpoint_keys = set(checkpoint.keys())
        not_loaded_keys = checkpoint_keys - loaded_keys

        del checkpoint

        # Process keys to log only first-level names
        loaded_first_level_keys = {key.split('.')[0] for key in loaded_keys}
        not_loaded_first_level_keys = {
            key.split('.')[0] for key in not_loaded_keys}

        # Log unique first-level keys
        log.info(f"Loaded first-level keys: {sorted(loaded_first_level_keys)}")
        log.info(
            f"First-level keys not loaded: {sorted(not_loaded_first_level_keys)}")

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "encoder": [self.encoder],
            "sandwich": [self.encoder, self.downstream_head],
            "head": [self.downstream_head],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _encode_images(self, views, chunk_size=400):
        B = views[0]["img"].shape[0]

        encoded_feats, shapes = [], []
        for view in views:
            img = view["img"]
            true_shape = view.get(
                "true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1)
            )

            # CNN encoder returns spatial features directly
            feat = self.encoder(img)  # B x C x H x W
            # feat = feat.view(
            #     B, feat.shape[1], -1).transpose(1, 2)
            encoded_feats.append(feat)
            shapes.append(true_shape)

        return encoded_feats, shapes  # No positions needed

    def set_max_parallel_views_for_head(self, max_parallel_views_for_head):
        # expose this to user to control the number of views processed in parallel in the head
        self.max_parallel_views_for_head = max_parallel_views_for_head

    def forward(self, views, profiling=False):
        """
        Args:
            views (list[dict]): a list of views, each view is a dict of tensors, the tensors are batched

        Returns:
            list[dict]: a list of results for each view
            dict: profiling information (if profiling=True)
        """
        # Initialize profiling dict
        profiling_info = {} if profiling else None

        # encode the images --> B,S,D
        encode_images_start_time = time.time()
        
        # encoded_feats, positions, shapes = self._encode_images(views)
        encoded_feats, shapes = self._encode_images(views)
        # print(f"Encoded features shape: ")
        # for i, encoded_feat in enumerate(encoded_feats): 
        #     print(f"encoded_feats[{i}].shape: {encoded_feat.shape}")  # 1,768,1024
        encode_images_end_time = time.time()
        if profiling:
            torch.cuda.synchronize()
            encode_images_end_time = time.time()
            encode_time = encode_images_end_time - encode_images_start_time
            profiling_info["encode_images_time"] = encode_time
            print(f"encode_images time: {encode_time}")
        if encode_images_end_time - encode_images_start_time > 20:
            print(
                f"something is wrong with the encoder, it took: {encode_images_end_time - encode_images_start_time}")
            # print the image and true_shape
            for view_idx, view in enumerate(views):
                print(
                    f"view_idx: {view_idx}\n, view name: {view_name(view)}\n, image content: {view['img']}\n, true_shape: {view['true_shape']}")

        # Create image IDs for each patch
        pos_emb_start_time = time.time()
        num_images = len(views)

        # B, _, _ = encoded_feats[0].shape
        B = len(encoded_feats[0])

        different_resolution_across_views = not all(
            torch.equal(shapes[0], shape) for shape in shapes)

        # Initialize an empty list to collect image IDs for each patch.
        # Note that at inference time, different views may have different number of patches.
        image_ids = []

        # Loop through each encoded feature to get the actual number of patches
        for i, encoded_feat in enumerate(encoded_feats):
            # Get the number of patches for this image
            num_patches = encoded_feat.shape[1]
            # print(f"Image {i} has {num_patches} patches")
            # Extend the image_ids list with the current image ID repeated num_patches times
            image_ids.extend([i] * num_patches)
            # print(f"Current image_ids length: {len(image_ids)}")

        # print(f"Total image_ids length: {len(image_ids)}")
        # print(f"Image IDs: {image_ids}")

        # Repeat the image_ids list B times and reshape it to match the expected shape
        image_ids = torch.tensor(
            image_ids * B).reshape(B, -1).to(encoded_feats[0].device)
        # print(f"image_ids shape: {image_ids.shape}")
        if profiling:
            pos_emb_time = time.time() - pos_emb_start_time
            profiling_info["pos_emb_time"] = pos_emb_time
            print(f"pos emb time: {pos_emb_time}")

        # combine all ref images into object-centric representation
        if profiling:
            torch.cuda.synchronize()
            decoder_start_time = time.time()
            
        dec_output = self.decoder(encoded_feats, image_ids)
        # print(f"dec_output shape: ")
        # for i, dec_out in enumerate(dec_output):
        #     print(f"dec_output[{i}].shape: {dec_out.shape}")

        if profiling:
            torch.cuda.synchronize()
            decoder_time = time.time() - decoder_start_time
            profiling_info["decoder_time"] = decoder_time
            print(f"decoder time: {decoder_time}")

        ################## Forward pass through the head ##################
        # TODO: optimize this

        # Initialize the final results list
        final_results = [{} for _ in range(num_images)]

        head_prepare_input_start_time = time.time()
        # Prepare the gathered outputs for each layer
        if different_resolution_across_views or self.training:
            # print("Different resolution across views or training mode")
            # Precompute the number of patches per image
            num_patches_list = [encoded_feat.shape[1]
                                for encoded_feat in encoded_feats]

            gathered_outputs_list = [[]
                                     for _ in range(num_images)]  # List per image
            for layer_output in dec_output:
                # layer_output: (B, P_total, D)
                # Split layer_output along dimension 1 according to num_patches_list
                split_layer_outputs = torch.split(
                    layer_output, num_patches_list, dim=1)
                for img_id, gathered_output in enumerate(split_layer_outputs):
                    # gathered_output: (B, num_patches_list[img_id], D)
                    gathered_outputs_list[img_id].append(gathered_output)
        else:
            # print("Same resolution across views")
            # All images have the same number of patches
            P_patches = encoded_feats[0].shape[1]
            gathered_outputs_list = []
            
            for layer_output in dec_output:
                # layer_output: (B, num_images * P_patches, D)
                # Rearrange to (num_images * B, P_patches, D)
                layer_output = rearrange(
                    layer_output,
                    'B (num_images P_patches) D -> (num_images B) P_patches D',
                    num_images=num_images,
                    P_patches=P_patches
                )
                # print(f"layer_output shape after rearrange: {layer_output.shape}")
                gathered_outputs_list.append(layer_output)

        if profiling:
            head_prepare_input_time = time.time() - head_prepare_input_start_time
            profiling_info["head_prepare_input_time"] = head_prepare_input_time
            print(f"head prepare input time: {head_prepare_input_time}")

        head_forward_start_time = time.time()
        with profiler.record_function("head: forward pass"):
            # print("Forward pass through the head")
            if different_resolution_across_views or self.training:
                # print("Processing views sequentially due to different resolutions or training mode")
                # If the views have different resolutions, we cannot batch the views together
                # or if we are in training mode, we can batch the views together, but we dont want to get OOM so we process them sequentially
                # Forward pass for each view separately
                final_results = [{} for _ in range(num_images)]
                for img_id in range(num_images):
                    img_result = self.head(
                        gathered_outputs_list[img_id], shapes[img_id])
                    # print(f"img_result shape for image {img_id}: {img_result['pts3d'].shape}")
                    if self.local_head:
                        local_img_result = self.local_head(
                            gathered_outputs_list[img_id], shapes[img_id])
                        # print(f"local_img_result shape for image {img_id}: {local_img_result['pts3d'].shape}")

                    # Re-map the results back to the original batch and image order
                    for key in img_result.keys():
                        if key == 'pts3d':
                            final_results[img_id]['pts3d_in_other_view'] = img_result[key]
                        else:
                            final_results[img_id][key] = img_result[key]

                    # Store local head output if available
                    if self.local_head:
                        final_results[img_id]['pts3d_local'] = local_img_result['pts3d']
                        if 'conf' in local_img_result:
                            final_results[img_id]['conf_local'] = local_img_result['conf']
            else:  # if we are in inference mode and all views have the same resolution, we can batch the views together
                # print("Processing views in parallel due to same resolution")
                concatenated_shapes = torch.cat(shapes, dim=0)

                # Split concatenated_shapes into chunks outside the loop
                shape_chunks = torch.split(
                    concatenated_shapes, self.max_parallel_views_for_head, dim=0)
                # Determine number of chunks from shape_chunks
                num_chunks = len(shape_chunks)

                # Initialize a list to hold chunked gathered outputs
                chunked_gathered_outputs_list = [[] for _ in range(num_chunks)]

                # Split gathered_outputs_list into chunks
                for layer_output in gathered_outputs_list:
                    # Split the layer_output along (num_images * B) dimension
                    split_layer_outputs = torch.split(
                        layer_output, self.max_parallel_views_for_head, dim=0)
                    # print(f"Number of split_layer_outputs: {len(split_layer_outputs)}")
                    # print(f"Shape of each split_layer_output: {split_layer_outputs[0].shape}")
                    for chunk_idx, split_output in enumerate(split_layer_outputs):
                        chunked_gathered_outputs_list[chunk_idx].append(
                            split_output)
                        # print(f"chunked_gathered_outputs_list[{chunk_idx}] shape: {split_output.shape}")

                # Initialize lists to hold results for each chunk
                result_chunks = []
                local_result_chunks = [] if self.local_head else None

                # Process each chunk through self.head and local_head
                for chunk, chunk_shapes in zip(chunked_gathered_outputs_list, shape_chunks):
                    # Forward pass for self.head
                    result_chunk = self.head(chunk, chunk_shapes)
                    result_chunks.append(result_chunk)

                    # Forward pass for local head if available
                    if self.local_head:
                        local_result_chunk = self.local_head(
                            chunk, chunk_shapes)
                        local_result_chunks.append(local_result_chunk)

                # Reassemble chunks
                result = {key: torch.cat(
                    [chunk[key] for chunk in result_chunks], dim=0) for key in result_chunks[0].keys()}

                if self.local_head:
                    local_result = {key: torch.cat(
                        [chunk[key] for chunk in local_result_chunks], dim=0) for key in local_result_chunks[0].keys()}

                # Re-map the results from num_images * B tensor to list of B tensors
                # Initialize the final results list
                final_results = [{} for _ in range(num_images)]

                # Re-map the results back to the original batch and image order
                for key in result.keys():
                    for img_id in range(num_images):
                        img_result = result[key][img_id * B:(img_id + 1) * B]
                        if key == 'pts3d':
                            final_results[img_id]['pts3d_in_other_view'] = img_result
                        else:
                            final_results[img_id][key] = img_result

                        # Store local head output if available
                        if self.local_head:
                            local_img_result = local_result['pts3d'][img_id * B:(
                                img_id + 1) * B]
                            final_results[img_id]['pts3d_local'] = local_img_result
                            if 'conf' in local_result:
                                final_results[img_id]['conf_local'] = local_result['conf'][img_id * B:(
                                    img_id + 1) * B]
        if profiling:
            torch.cuda.synchronize()
            end_time = time.time()
            profiling_info["head_forward_time"] = end_time - \
                head_forward_start_time
            print(f"head forward time: {end_time - head_forward_start_time}")
            profiling_info["total_time"] = end_time - encode_images_start_time
            print(
                f"total Fast3R forward time: {end_time - encode_images_start_time}")

        # for key in final_results[0].keys():
        #     if isinstance(final_results[0][key], torch.Tensor):
        #         print(f"final_results[{key}].shape: {final_results[0][key].shape}")
        if profiling:
            return final_results, profiling_info
        else:
            return final_results


class LightDarkNet53(nn.Module):
    def __init__(self, embed_dim=768):
        super(LightDarkNet53, self).__init__()
        self.embed_dim = embed_dim
        self.backbone = timm.create_model(
            'cspdarknet53',
            pretrained=True,
            features_only=True
        )

        # Freeze backbone if specified
        self.feature_info = self.backbone.feature_info
        self.conv = nn.Conv2d(
            512, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        """
        Forward pass through the encoder
        Returns multi-scale features from different stages
        """
        x1 = self.backbone(image)[-2]
        x2 = self.conv(x1)
        features = x2.view(x2.shape[0], -1, x2.shape[1])
        print(features.shape)
        return features

    def get_feature_dims(self):
        """Get the number of channels at each stage"""
        return [info['num_chs'] for info in self.feature_info]


class Mambavision(nn.Module):
    def __init__(self, embed_dim=768):
        super(Mambavision, self).__init__()
        self.embed_dim = embed_dim
        self.backbone = AutoModel.from_pretrained(
            "nvidia/MambaVision-S-1K", trust_remote_code=True)

        # Freeze backbone if specified
        # self.feature_info = self.backbone.feature_info
        self.conv = nn.Conv2d(96, 1024, kernel_size=7, stride=4, padding=2)

    def forward(self, image):
        """
        Forward pass through the encoder
        Returns multi-scale features from different stages
        """
        x1 = self.backbone(image)[1][0]
        x2 = self.conv(x1)  # [1, 1024, 32, 24]
        features = x2.view(x2.shape[0], -1, x2.shape[1])
        # print(features.shape)
        return features

    def get_feature_dims(self):
        """Get the number of channels at each stage"""
        return [info['num_chs'] for info in self.feature_info]


class Efficient(nn.Module):
    def __init__(self, embed_dim=768):
        super(Efficient, self).__init__()
        self.embed_dim = embed_dim
        self.backbone = EfficientNet.from_pretrained('efficientnet-b6')

        self.conv = nn.Conv2d(200, self.embed_dim,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        """
        Forward pass through the encoder
        Returns multi-scale features from different stages
        """
        x1 = self.backbone.extract_endpoints(
            image)['reduction_4']  # [1, 200, 32, 24]
        x2 = self.conv(x1)  # [1, 1024, 32, 24]
        features = x2.view(x2.shape[0], -1, x2.shape[1])
        # print(features.shape)
        return features  # torch.Size([1, 768, 1024])

    def get_feature_dims(self):
        """Get the number of channels at each stage"""
        return [info['num_chs'] for info in self.feature_info]


class EfficientV2(nn.Module):
    def __init__(self, embed_dim=768):
        super(EfficientV2, self).__init__()
        self.embed_dim = embed_dim
        self.backbone = torchvision.models.efficientnet_v2_s(pretrained=True)

        self.conv = nn.Conv2d(1280, self.embed_dim,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        """
        Forward pass through the encoder
        Returns multi-scale features from different stages
        """
        x1 = self.backbone.features(image)  # [1, 1280, 16, 12]
        x1_d1, x1_d2 = x1.shape[-2:]
        x2 = F.interpolate(x1, size=(x1_d2*2, x1_d1*2),  # [1, 1280, 32, 24]
                           mode='bilinear',
                           align_corners=False)  # [1, 1280, 32, 24]
        # import ipdb; ipdb.set_trace()
        x3 = self.conv(x2)  # [1, 1024, 32, 24]
        features = x3.view(x3.shape[0], -1, x3.shape[1])
        # print(features.shape)
        return features  # torch.Size([1, 768, 1024])

    def get_feature_dims(self):
        """Get the number of channels at each stage"""
        return [info['num_chs'] for info in self.feature_info]


class MobileNetV4(nn.Module):
    def __init__(self, embed_dim=768):
        super(MobileNetV4, self).__init__()
        self.embed_dim = embed_dim
        self.backbone = timm.create_model('mobilenetv4_conv_large.e500_r256_in1k',
                                          pretrained=True, features_only=True)

        self.conv = nn.Conv2d(192, self.embed_dim,
                              kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        """
        Forward pass through the encoder
        Returns multi-scale features from different stages
        """
        x1 = self.backbone(image)[-2]  # [1, 192, 32, 24]
        # import ipdb; ipdb.set_trace()
        x2 = self.conv(x1)  # [1, 1024, 32, 24]
        features = x2.view(x2.shape[0], -1, x2.shape[1])
        # print(features.shape)
        return features  # torch.Size([1, 768, 1024])

    def get_feature_dims(self):
        """Get the number of channels at each stage"""
        return [info['num_chs'] for info in self.feature_info]


# class MobileNetV4_167(nn.Module):
#     def __init__(self, embed_dim=768):
#         super(MobileNetV4_167, self).__init__()
#         self.embed_dim = embed_dim
#         self.backbone = timm.create_model(
#             'mobilenetv4_conv_large.e500_r256_in1k',
#             pretrained=True, features_only=True)

#         self.conv = nn.Conv2d(192, int(self.embed_dim),
#                               kernel_size=3, stride=1, padding=1)
#         self.bn = nn.BatchNorm2d(int(self.embed_dim))

#     def forward(self, image, image167):
#         """
#         Forward pass through the encoder
#         Returns multi-scale features from different stages
#         """
#         x1 = self.backbone(image)[-2]  # [1, 192, 32, 24]
#         x1_167 = self.backbone(image167)[-2]  # [1, 192, 32, 24]
#         # print(image.shape, image167.shape)
#         # print(x1.shape, x1_167.shape)
#         # import ipdb; ipdb.set_trace()
#         x2 = self.conv(x1)  # [1, 1024, 32, 24]
#         x2 = self.bn(x2)  # [1, 1024, 32, 24]
#         x2_167 = self.conv(x1_167)  # [1, 1024, 32, 24]
#         x2_167 = self.bn(x2_167)  # [1, 1024, 32, 24]
#         # Concatenate the features from both images along the channel dimension
#         # x2_mix = torch.cat((x2, x2_167), dim=1)
#         x2_mix = x2 + x2_167  # [1, 1024, 32, 24]
#         features = x2_mix.view(x2_mix.shape[0], -1, x2_mix.shape[1])
#         # print(features.shape)
#         return features  # torch.Size([1, 768, 1024])

#     def get_feature_dims(self):
#         """Get the number of channels at each stage"""
#         return [info['num_chs'] for info in self.feature_info]


class MobileNetV4_167(nn.Module):
    def __init__(self, embed_dim=768):
        super(MobileNetV4_167, self).__init__()
        self.embed_dim = embed_dim
        self.backbone = timm.create_model(
            'mobilenetv4_conv_large.e500_r256_in1k',
            pretrained=True, features_only=True)

        self.conv2 = nn.Conv2d(192, int(self.embed_dim//4),
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(self.embed_dim//4))
        self.conv3 = nn.Conv2d(96, int(self.embed_dim//4),
                              kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(int(self.embed_dim//4))
        self.conv4 = nn.Conv2d(48, int(self.embed_dim//4),
                              kernel_size=6, stride=4, padding=1)
        self.bn4 = nn.BatchNorm2d(int(self.embed_dim//4))
        self.conv5 = nn.Conv2d(24, int(self.embed_dim//4),
                              kernel_size=10, stride=8, padding=1)
        self.bn5 = nn.BatchNorm2d(int(self.embed_dim//4))
        
    def forward(self, image, image167):
        """
        Forward pass through the encoder
        Returns multi-scale features from different stages
        """
        # torch.Size([1, 24, 192,256])
        # torch.Size([1, 48, 96, 128])
        # torch.Size([1, 96, 48, 64])
        # torch.Size([1, 192, 24, 32])
        # torch.Size([1, 960, 12, 16])
        # [print(aa.shape) for aa in self.backbone(image)]
        x2 = self.backbone(image)[-2]  # [1, 192, 24, 32]
        x2_167 = self.backbone(image167)[-2]  # [1, 192, 24, 32]
        
        x3 = self.backbone(image)[-3]  # [1,96,48,64]
        x3_167 = self.backbone(image167)[-3]  # [1,96,48,64]
        
        x4 = self.backbone(image)[-4]  # [1, 48, 96, 128]
        x4_167 = self.backbone(image167)[-4]  # [1, 48, 96, 128]
        
        x5 = self.backbone(image)[-5]  # [1, 24, 192, 256]
        x5_167 = self.backbone(image167)[-5]  # [1, 24, 192, 256]
        # print(image.shape, image167.shape)
        # print(x1.shape, x1_167.shape)
        # import ipdb; ipdb.set_trace()
        x2_c = self.bn2(self.conv2(x2))  # [1, 256, 32, 24]
        x2_167_c = self.bn2(self.conv2(x2_167))  # [1, 256, 32, 24]
        x3_c = self.bn3(self.conv3(x3))  # [1, 256, 32, 24]
        x3_167_c = self.bn3(self.conv3(x3_167))  # [1, 256, 32, 24]
        x4_c = self.bn4(self.conv4(x4))  # [1, 256, 32, 24]
        x4_167_c = self.bn4(self.conv4(x4_167))  # [1, 256, 32, 24]
        x5_c = self.bn5(self.conv5(x5))  # [1, 256, 32, 24]
        x5_167_c = self.bn5(self.conv5(x5_167))  # [1, 256, 32, 24]
        # Concatenate the features from both images along the channel dimension
        # import ipdb; ipdb.set_trace()
        # x2_mix = torch.cat((x2, x2_167), dim=1)
        x2_mix = x2_c + x2_167_c  # [1,256, 32, 24]
        x3_mix = x3_c + x3_167_c  # [1,256, 32, 24]
        x4_mix = x4_c + x4_167_c  # [1,256, 32, 24]
        x5_mix = x5_c + x5_167_c  # [1,256, 32, 24]
        
        # Concatenate all features along the channel dimension
        # print(f"x2_mix shape: {x2_mix.shape}, x3_mix shape: {x3_mix.shape}, x4_mix shape: {x4_mix.shape}, x5_mix shape: {x5_mix.shape}")
        x2345_mix = torch.cat((x2_mix, x3_mix, x4_mix, x5_mix), dim=1)  # [1, 1024, 32, 24]
        features = x2345_mix.view(x2345_mix.shape[0], -1, x2345_mix.shape[1])
        # print(features.shape)
        return features  # torch.Size([1, 768, 1024])

    def get_feature_dims(self):
        """Get the number of channels at each stage"""
        return [info['num_chs'] for info in self.feature_info]

class EfficientnetV2_L(nn.Module):
    def __init__(self, embed_dim=1024):
        super(EfficientnetV2_L, self).__init__()
        self.embed_dim = embed_dim
        self.backbone = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', 
                                          pretrained=True,features_only=True)

        self.pool8 = torch.nn.MaxPool2d(8,8)
        self.pool4 = torch.nn.MaxPool2d(4,4)
        self.pool2 = torch.nn.MaxPool2d(2,2)
        
        self.conv0 = nn.Conv2d(32, int(self.embed_dim//4),
                              kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(64, int(self.embed_dim//4),
                              kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, int(self.embed_dim//4),
                              kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(224, int(self.embed_dim//4),
                              kernel_size=3, stride=1, padding=1)
        
        self.bn0 = nn.BatchNorm2d(int(self.embed_dim//4))
        self.bn1 = nn.BatchNorm2d(int(self.embed_dim//4))
        self.bn2 = nn.BatchNorm2d(int(self.embed_dim//4))
        self.bn3 = nn.BatchNorm2d(int(self.embed_dim//4))
        
        self.gelu = nn.GELU()
        

    def forward(self, image):
        """
        Forward pass through the encoder
        Returns multi-scale features from different stages
        
        """
        # >>> a1[0].shape
        # torch.Size([1, 32, 256, 192])
        # >>> a1[1].shape
        # torch.Size([1, 64, 128, 96])
        # >>> a1[2].shape
        # torch.Size([1, 96, 64, 48])
        # >>> a1[3].shape
        # torch.Size([1, 224, 32, 24])
        # >>> a1[4].shape
        # torch.Size([1, 640, 16, 12])
        
        x_all = self.backbone(image)
        x0 = x_all[0]  # [1, 32, 256, 192]
        x1 = x_all[1]  # [1, 64, 128, 96]
        x2 = x_all[2]  # [1, 96, 64, 48]
        x3 = x_all[3]  # [1, 224, 32, 24]
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # import ipdb;
        # ipdb.set_trace()
        x01 = x0.reshape(x0.shape[0], -1, x3.shape[-2], x3.shape[-1])  # [1, 32*64, 32, 24] 2048
        x11 = x1.reshape(x1.shape[0], -1, x3.shape[-2], x3.shape[-1])  # [1, 64*16, 32, 24] 1024
        x21 = x2.reshape(x2.shape[0], -1, x3.shape[-2], x3.shape[-1])  # [1, 96*4, 32, 24] 384
        x31 = x3  # [1, 224, 32, 24]
        x2 = torch.cat((x01, x11, x21, x31), dim=1)  # [1, 2048+1024+384+224, 32, 24] = [1, 3680, 32, 24]
        
        # x00 = self.gelu(self.bn0(self.conv0(x0)))  # [1, 256, 256, 192]
        # x00p = self.pool8(x00)  # [1, 256, 32, 24]
        # x01 = self.gelu(self.bn1(self.conv1(x1)))  # [1, 256, 128, 96]
        # x01p = self.pool4(x01)  # [1, 256, 32, 24]
        # x02 = self.gelu(self.bn2(self.conv2(x2)))  # [1, 256, 64, 48]
        # x02p = self.pool2(x02)  # [1, 256, 32, 24]
        # x03 = self.gelu(self.bn3(self.conv3(x3)))  # [1, 256, 32, 24]
        # x2 = torch.cat((x00p, x01p, x02p, x03), dim=1)  # [1, 256*4, 32, 24] = [1, 1024, 32, 24]
        
        # x_cat = torch.cat((x01, x11, x21, x31), dim=1)  # [1, 32*64+64*16+96*4+224, 32, 24] = [1, 2048+864+384+224, 32, 24] = [1, 3520, 32, 24]
        
        # x2 = self.conv(x_cat)  # [1, 1024, 32, 24]
        features = x2.view(x2.shape[0], -1, x2.shape[1]) # [1, 768, 3680]
        # print(features.shape)
        return features  # torch.Size([1, 768, 1024])

    def get_feature_dims(self):
        """Get the number of channels at each stage"""
        return [info['num_chs'] for info in self.feature_info]
    
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
    ):
        super(ResNetEncoder, self).__init__()

        # Store original parameters for compatibility
        self.embed_dim = embed_dim

        # Replace transformer with ResNet18
        self.backbone = timm.create_model(
            'resnet18', pretrained=True, features_only=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.conv = nn.Conv2d(256, embed_dim, stride=1,
                              padding=1, kernel_size=3)

    def forward(self, image):
        # Get ResNet features
        x1 = self.backbone(image)[-2]
        x2 = self.conv(x1)
        features = x2.view(x2.shape[0], -1, x2.shape[1])
        return features


class MambaFusionDecoder(nn.Module):
    def __init__(self, enc_embed_dim, embed_dim=768, num_layers=12):
        super().__init__()
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim)

        # # Mamba blocks for multi-view fusion
        # self.mamba_blocks = nn.ModuleList([
        #     BiMamba2Block(embed_dim, state_size=16)
        #     for _ in range(num_layers)
        # ])

        # # Cross-view fusion with Mamba
        # self.cross_mamba = CrossMamba(embed_dim)
        # self.final_norm = RMSNorm(embed_dim)


class Fast3RDecoderCNN2(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        image_idx_emb_dim: int,
        embed_dim: int,
        enc_embed_dim: int,
        depth: int,
    ):
        super(Fast3RDecoderCNN2, self).__init__()

        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        # # initialize the positional embedding for the decoder
        self.random_image_idx_embedding = random_image_idx_embedding

        # final norm layer
        # self.dec_norm = norm_layer(embed_dim)
        self.num_outputs = depth + 1
        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(1000))
            ).float(),
            persistent=False,
        )

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=True),
                nn.ReLU6()
            ) for _ in range(depth)
        ])

        # self.projections = nn.Sequential(
        #     nn.Linear(embed_dim, depth*embed_dim, bias=True),
        #     nn.ReLU6()
        # )

    def _generate_per_rank_generator(self):
        # this way, the randperm will be different for each rank, but deterministic given a fixed number of forward passes (tracked by self.random_generator)
        # and to ensure determinism when resuming from a checkpoint, we only need to save self.random_generator to state_dict
        # generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank(
        ) if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """
        # Generate random non-repeating image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank, while also being deterministic based on the global across numbers of forward passes
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random non-repeating IDs for the remaining views using the generator
        for b in range(batch_size):
            # Use the torch.Generator for randomness to ensure randomness between forward passes
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[
                :num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)

        # Initialize list to store positional embeddings for all views
        image_pos_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            image_pos_for_view = image_pos_for_view.unsqueeze(
                1).repeat(1, num_patches, 1)

            image_pos_list.append(image_pos_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        image_pos = torch.cat(image_pos_list, dim=1)  # (B, Npatches_total, D)

        return image_pos

    def forward(self, encoded_feats, image_ids):
        """ Forward pass through the decoder.

        Args:
            encoded_feats (list of tensors): Encoded features for each view. Shape: B x Npatches x D
            positions (list of tensors): Positional embeddings for each view. Shape: B x Npatches x 2
            image_ids (tensor): Image IDs for each patch. Shape: B x Npatches
        """
        x = torch.cat(
            encoded_feats, dim=1)  # concate along the patch dimension
        
        # print(f"Decoder input x shape: {x.shape}")

        final_output = [x]  # before projection

        # project to decoder dim
        x = self.decoder_embed(x)

        # Add positional embedding based on image IDs
        if self.random_image_idx_embedding:
            # Generate random positional embeddings for all views and samples
            image_pos = self._get_random_image_pos(
                encoded_feats=encoded_feats,
                batch_size=encoded_feats[0].shape[0],
                num_views=len(
                    encoded_feats),
                max_image_idx=self.image_idx_emb.shape[0] - 1,
                device=x.device)
        else:
            # Use default image IDs from input
            num_images = (torch.max(image_ids) + 1).cpu().item()
            image_idx_emb = self.image_idx_emb[:num_images]
            image_pos = image_idx_emb[image_ids]
        # print(f"image_pos {image_pos.shape}")
        # Apply positional embedding based on image IDs and positions

        x += image_pos  # x has size B x Npatches x D, image_pos has size Npatches x D, so this is broadcasting

        for blk in self.projections:
            x1 = blk(x)
            final_output.append(x1)
            # print(x1.shape)
        # output = self.projections(x)  # (B, Npatches, D * depth)
        # import ipdb; ipdb.set_trace()
        # output = output.view(
        #     output.shape[0], output.shape[1], self.num_outputs, -1)
        # output = output.permute(0, 2, 1, 3)

        return final_output


class Fast3RDecoderCNN(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        enc_embed_dim: int,
        embed_dim: int = 768,
        num_cnn_layers: int = 6,  # Replace depth parameter
        kernel_size: int = 3,     # New parameter for CNN
        stride: int = 1,          # New parameter for CNN
        padding: int = 1,         # New parameter for CNN
        norm_layer=nn.BatchNorm2d,  # Use BatchNorm2d for CNN
        num_images: int = 10
    ):
        super(Fast3RDecoderCNN, self).__init__()

        # transfer from encoder to decoder

        # Replace Linear projection with Conv2d for spatial features
        # Replace transformer blocks with CNN blocks
        # self.conv_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(embed_dim, embed_dim, kernel_size,
        #                   stride=stride, padding=padding),
        #         norm_layer(embed_dim),
        #         nn.ReLU6(inplace=True),
        #         nn.Conv2d(embed_dim, embed_dim, kernel_size,
        #                   stride=stride, padding=padding),
        #         norm_layer(embed_dim),
        #         nn.ReLU6(inplace=True),
        #         # Optional: Add residual connection
        #         # 1x1 conv for residual
        #         nn.Conv2d(embed_dim, embed_dim, kernel_size=3)
        #     ) for _ in range(num_cnn_layers)
        # ])
        self.num_cnn_layers = num_cnn_layers
        self.conv_block_0 = nn.Sequential(
            nn.Conv2d(enc_embed_dim*num_images, enc_embed_dim, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            norm_layer(enc_embed_dim),
            nn.ReLU6(inplace=True))
        # self.conv_block_1 = nn.Sequential(
        #     nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images, kernel_size=kernel_size,
        #               stride=stride, padding=padding),
        #     norm_layer(enc_embed_dim),
        #     nn.ReLU6(inplace=True))
        # self.conv_blocks = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images, kernel_size=kernel_size,
        #                   stride=stride, padding=padding),
        #         norm_layer(enc_embed_dim),
        #         nn.ReLU6(inplace=True),
        #     ) for _ in range(num_cnn_layers - 1)])

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size,
                      stride=stride, padding=padding),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size,
                      stride=stride, padding=padding),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size,
                      stride=stride, padding=padding),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size+2,
                      stride=stride, padding=padding+1),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size+2,
                      stride=stride, padding=padding+1),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size+2,
                      stride=stride, padding=padding+1),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size+4,
                      stride=stride, padding=padding+2),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size+4,
                      stride=stride, padding=padding+2),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_9 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size+4,
                      stride=stride, padding=padding+2),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        self.conv_block_10 = nn.Sequential(
            nn.Conv2d(enc_embed_dim, enc_embed_dim*num_images,
                      kernel_size=kernel_size+4,
                      stride=stride, padding=padding+2),
            norm_layer(enc_embed_dim * num_images),
            nn.ReLU6(inplace=True))

        # Keep image position embeddings (this is the key part you want to preserve)
        self.random_image_idx_embedding = random_image_idx_embedding
        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(1000))
            ).float(),
            persistent=False,
        )

        # Replace LayerNorm with BatchNorm2d for CNN features
        self.final_norm = norm_layer(enc_embed_dim*num_images)

    def _generate_per_rank_generator(self):
        # this way, the randperm will be different for each rank, but
        # deterministic given a fixed number of forward passes
        # (tracked by self.random_generator)
        # and to ensure determinism when resuming from a checkpoint, we only
        # need to save self.random_generator to state_dict
        # generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank(
        ) if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """
        # Generate random non-repeating image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank, while also being deterministic based on the global across numbers of forward passes
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random non-repeating IDs for the remaining views using the generator
        for b in range(batch_size):
            # Use the torch.Generator for randomness to ensure randomness between forward passes
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[
                :num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)

        # Initialize list to store positional embeddings for all views
        image_pos_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            image_pos_for_view = image_pos_for_view.unsqueeze(
                1).repeat(1, num_patches, 1)

            image_pos_list.append(image_pos_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        image_pos = torch.cat(image_pos_list, dim=1)  # (B, Npatches_total, D)

        return image_pos

    def forward(self, encoded_feats, image_ids):
        """ CNN-based decoder that only uses image position embeddings.

        Args:
            encoded_feats (list of tensors): CNN features for each view. Shape: B x C x H x W
            image_ids (tensor): Image IDs for each spatial location
        """

        # Process CNN features directly without converting to patches
        processed_feats = []
        for feat in encoded_feats:
            if len(feat.shape) == 4:  # B x C x H x W
                processed_feats.append(feat)
            else:  # Convert back to spatial if needed
                B, N, C = feat.shape
                H = W = int(N ** 0.5)  # Assume square feature maps
                feat = feat.transpose(1, 2).view(B, C, H, W)
                processed_feats.append(feat)

        # Concatenate features along channel dimension or spatially
        if len(processed_feats) > 1:
            x = torch.cat(processed_feats, dim=1)  # B x (C*num_views) x H x W
        else:
            x = processed_feats[0]

        # Add only image position embeddings (no spatial positional embeddings)
        if self.random_image_idx_embedding:
            image_pos = self._get_random_image_pos(
                encoded_feats=processed_feats,
                batch_size=x.shape[0],
                num_views=len(processed_feats),
                max_image_idx=self.image_idx_emb.shape[0] - 1,
                device=x.device
            )
        else:
            num_images = (torch.max(image_ids) + 1).cpu().item()
            # Shape: num_images x D
            image_idx_emb = self.image_idx_emb[:num_images]

            # Broadcast image embeddings to spatial dimensions
            B, C, H, W = x.shape
            # Get embeddings for each location
            image_pos = image_idx_emb[image_ids]

            # Reshape to match spatial dimensions if needed
            if len(image_pos.shape) == 2:  # If image_ids is flattened
                image_pos = image_pos.view(
                    B, H, W, -1).permute(0, 3, 1, 2)  # B x D x H x W

        # Replace transformer blocks with CNN blocks

        # x = self.decoder_embed(x)  # Project to decoder dimension
        # import ipdb
        # ipdb.set_trace()

        B, C, H, W = x.shape
        print(f"image_pos {image_pos.shape}")
        if len(image_pos.shape) < len(x.shape):  # (B, embed_dim)
            # (B, embed_dim, 1, 1)
            image_pos = image_pos.unsqueeze(-1)  # 1, 2560, 192, 1

        # Now check and interpolate if needed
        # x = 1, 2560, 12, 16
        print(f"cnn_decoder {image_pos.shape}")
        if image_pos.shape[-2:] != x.shape[-2:]:
            # import ipdb; ipdb.set_trace()
            # Change to match spatial dimensions
            image_pos = F.interpolate(
                image_pos, size=(H, W), mode='bilinear', align_corners=False)

        # Add image position embeddings
        x = x + image_pos

        # CNN processing blocks
        # x = self.decoder_embed(x)

        # import ipdb
        # ipdb.set_trace()
        # x = self.cnn_decoder(x, image_pos)
        final_output = [x]  # 2560,192
        x = self.conv_block_0(x)  # Initial convolution block

        final_output.append(self.conv_block_1(x))
        final_output.append(self.conv_block_2(x))
        final_output.append(self.conv_block_3(x))
        final_output.append(self.conv_block_4(x))
        final_output.append(self.conv_block_5(x))
        final_output.append(self.conv_block_6(x))
        final_output.append(self.conv_block_7(x))
        final_output.append(self.conv_block_8(x))
        final_output.append(self.conv_block_9(x))
        final_output.append(self.conv_block_10(x))
        # Final normalization (can be BatchNorm instead of LayerNorm)

        return final_output

    # def cnn_decoder(self, x, image_pos):
    #     """ CNN-based decoder blocks replacing transformer attention """

    #     # Add image position embeddings
    #     if image_pos.shape[-2:] == x.shape[-2:]:  # Same spatial dimensions
    #         x = x + image_pos
    #     else:
    #         # Interpolate image_pos to match x dimensions if needed
    #         image_pos = F.interpolate(
    #             image_pos, size=x.shape[-2:], mode='bilinear',
    #             align_corners=False)
    #         x = x + image_pos

    #     # CNN processing blocks (replace transformer blocks)
    #     for conv_block in self.conv_blocks:
    #         x = conv_block(x)

    #     return x
    def cnn_decoder(self, x, image_pos):
        """ CNN-based decoder blocks replacing transformer attention """

        B, C, H, W = x.shape  # 1, 10240, 24, 32
        # import ipdb
        # ipdb.set_trace()
        # Fix: Always reshape image_pos to have spatial dimensions FIRST

        print(f"f1 {image_pos.shape}")
        if len(image_pos.shape) < len(x.shape):  # (B, embed_dim)
            # (B, embed_dim, 1, 1)
            image_pos = image_pos.unsqueeze(-1)  # 1, 10240, 768, 1

        # Now check and interpolate if needed
        # image_pos = 1,10240, 768
        # x = 1,10240, 24,32
        print(f"cnn_decoder {image_pos.shape}")
        if image_pos.shape[-2:] != x.shape[-2:]:
            # import ipdb; ipdb.set_trace()
            # Change to match spatial dimensions
            image_pos = F.interpolate(
                image_pos, size=(H, W), mode='bilinear', align_corners=False)

        # Add image position embeddings
        x = x + image_pos

        # CNN processing blocks
        x = self.decoder_embed(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.last_cnn_block(x)
        return x
