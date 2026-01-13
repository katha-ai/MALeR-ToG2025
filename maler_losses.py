import torch
import torch.nn.functional as F
import math

class MALeRLosses:
    def __init__(self, lambda_reg=0.0, reg_type=False, lambda_kl=0.0, kl_type=False, early_iterations=5, kl_threshold=0.001, verbose=False):
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.lambda_kl = lambda_kl
        self.kl_type = kl_type
        self.early_iterations = early_iterations
        self.kl_threshold = kl_threshold
        self.verbose = verbose

    def compute_outside_bbox_mask(self, latents, boxes):
        """
        Getting the mask for regions outside the bounding boxes
        """
        bbox_mask = torch.zeros_like(latents, dtype=torch.bool)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            y_min_idx, y_max_idx = int(y_min * latents.shape[-2]), int(y_max * latents.shape[-2])
            x_min_idx, x_max_idx = int(x_min * latents.shape[-1]), int(x_max * latents.shape[-1])
            bbox_mask[:, :, y_min_idx:y_max_idx, x_min_idx:x_max_idx] = 1
        return ~bbox_mask

    def compute_reg_loss(self, latents, prev_latents, outside_bbox_mask, i):
        """
        Computing MALeR's regularization loss
        """
        if prev_latents is not None and i < self.early_iterations:
            diff = (latents - prev_latents) * outside_bbox_mask
            reg_loss = (diff ** 2).sum() if self.reg_type else diff.abs().sum()
            if self.verbose:
                print(f"[LatentDiff] step={i}, reg_loss={reg_loss.item():.4f}")
            return self.lambda_reg * reg_loss
        return torch.tensor(0.0, device=latents.device)

    def compute_kl_loss(self, latents, i):
        """
        Computing MALeR's KL regularization loss
        """
        if i < self.early_iterations:
            mu = latents.mean(dim=1, keepdim=True)
            var = latents.var(dim=1, keepdim=True, unbiased=False)
            var = torch.clamp(var, min=1e-8)
            logvar = torch.log(var)
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - var)
            kl_loss = kl_loss.mean()
            kl_val = kl_loss.item()
            if kl_val > self.kl_threshold:
                if self.verbose:
                    print(f"Applying KL Loss: {kl_val:.4f}")
                return self.lambda_kl * kl_loss
            else:
                if self.verbose:
                    print(f"Skipping KL Loss, kl_loss={kl_val:.4f} <= {self.kl_threshold}")
                return torch.tensor(0.0, device=latents.device)
        return torch.tensor(0.0, device=latents.device)
    
    def compute_loss(self, latents, prev_latents, outside_bbox_mask, i):
        reg_loss = self.compute_reg_loss(latents, prev_latents, outside_bbox_mask, i)
        kl_loss = self.compute_kl_loss(latents, i)
        return reg_loss + kl_loss
    

    def compute_attribute_loss(
            self,
            subject_attn_maps,
            convert_boxes_to_masks_fn,
            subject_token_indices,
            sym_kl_weight,
            dissim_weight,
            device
    ):

        """
        Computing MALeR's subject attribute similarity and dissimilarity losses
        """
        #sim
        sim_loss_term, box_masks = compute_modifier_noun_similarity(
            subject_attn_maps,
            convert_boxes_to_masks_fn
        )
        
        # dissim
        dissim_loss = compute_modifier_noun_dissimilarity(
            subject_attn_maps,
            box_masks,
            subject_token_indices
        )

        subject_attn_maps.clear()

        if sim_loss_term is not None:
            total_sim_loss = sym_kl_weight * sim_loss_term
            total_dissim_loss = dissim_weight * dissim_loss
        else:
            total_sim_loss = torch.tensor(0.0, device=device)
            total_dissim_loss = torch.tensor(0.0, device=device)

        return (total_sim_loss + total_dissim_loss)
    


def compute_masked_symmetric_kl(attn1, attn2, box_mask=None, eps=1e-6):
    if box_mask is None:
        attn1_masked = attn1
        attn2_masked = attn2
    else:
        attn1_masked = attn1[..., box_mask]
        attn2_masked = attn2[..., box_mask]
    

    attn1_masked = torch.clamp(attn1_masked, min=eps)
    attn2_masked = torch.clamp(attn2_masked, min=eps)
    
    attn1_sum = attn1_masked.sum(dim=-1, keepdim=True)
    attn2_sum = attn2_masked.sum(dim=-1, keepdim=True)
    
    attn1_sum = torch.clamp(attn1_sum, min=eps)
    attn2_sum = torch.clamp(attn2_sum, min=eps)
    
    attn1_masked = attn1_masked / attn1_sum
    attn2_masked = attn2_masked / attn2_sum
    
    attn1_masked = attn1_masked + eps
    attn2_masked = attn2_masked + eps

    kl1 = F.kl_div(attn1_masked.log(), attn2_masked, reduction='batchmean')
    kl2 = F.kl_div(attn2_masked.log(), attn1_masked, reduction='batchmean')
    
    return 0.5 * (kl1 + kl2)



def compute_modifier_noun_dissimilarity(attn_maps, box_masks, subject_token_indices):
    subject_group_dissim = {}

    for current_group_idx, current_group in enumerate(subject_token_indices):
        if len(current_group) <= 1:  
            continue
            
        noun_idx = current_group[-1]
        modifier_indices = current_group[:-1]
        if not modifier_indices:
            continue


        for other_group_idx, other_group in enumerate(subject_token_indices):
            if other_group_idx == current_group_idx or len(other_group) <= 1:
                continue

            other_noun_idx = other_group[-1]
            other_box_mask = box_masks[other_group_idx]
            for modifier_idx in modifier_indices:
                modifier_found = False
                modifier_maps = None
                for key in attn_maps:
                    if isinstance(key[0], tuple) and len(key[0]) >= 3 and key[0][1] == modifier_idx and key[1] == current_group_idx:
                        modifier_key = key
                        modifier_maps = attn_maps[key]
                        modifier_found = True
                        break
                        

                if not modifier_found or not modifier_maps:
                    continue
                
                other_noun_found = False
                other_noun_maps = None
                for key in attn_maps:
                    if isinstance(key[0], tuple) and len(key[0]) >= 3 and key[0][2] == other_noun_idx and key[1] == other_group_idx:
                        other_noun_key = key
                        other_noun_maps = attn_maps[key]
                        other_noun_found = True
                        break
                        
                if not other_noun_found or not other_noun_maps:
                    continue
                
                if not modifier_maps['modifier'] or not other_noun_maps['noun']:
                    continue
                
                modifier_attn_combined = torch.stack(modifier_maps['modifier']).mean(dim=0)
                other_noun_combined = torch.stack(other_noun_maps['noun']).mean(dim=0)
                
                modifier_in_other_region = modifier_attn_combined[..., other_box_mask]
                other_noun_in_its_region = other_noun_combined[..., other_box_mask]

                symm_kl = compute_masked_symmetric_kl(
                    modifier_in_other_region,
                    other_noun_in_its_region,
                    None  
                )

                group_pair_key = (current_group_idx, other_group_idx)
                if group_pair_key not in subject_group_dissim:
                    subject_group_dissim[group_pair_key] = []

                subject_group_dissim[group_pair_key].append(symm_kl)

    if not subject_group_dissim:
        return torch.tensor(0.0, device=next(iter(attn_maps.values()))['modifier'][0].device if attn_maps else torch.device('cpu'))

    group_max_dissim = []
    for group_pair, kl_values in subject_group_dissim.items():
        if kl_values:
            sum_kl = sum(kl_values)
            group_max_dissim.append(sum_kl)
    
    if not group_max_dissim:
        return torch.tensor(0.0, device=next(iter(attn_maps.values()))['modifier'][0].device if attn_maps else torch.device('cpu'))
    
    dissim_loss = sum(group_max_dissim) / len(group_max_dissim)
    return (-dissim_loss)


def compute_modifier_noun_similarity(subject_attn_maps, convert_boxes_to_masks_fn, use_max=False):
    resolution = int(math.sqrt(next(iter(subject_attn_maps.values()))['modifier'][0].shape[-1]))
    box_masks = convert_boxes_to_masks_fn(resolution).flatten(start_dim=1)

    kl_losses = []
    subject_group_losses = {}

    for ((subject_group, modifier_idx, noun_idx), mask_idx), maps in subject_attn_maps.items():
        if maps['layer_count'] > 0:
            modifier_combined = torch.stack(maps['modifier']).mean(dim=0)
            noun_combined = torch.stack(maps['noun']).mean(dim=0)
            
            box_mask = box_masks[mask_idx]

            symm_kl = compute_masked_symmetric_kl(
                modifier_combined, 
                noun_combined,
                box_mask
            )

            subject_key = (subject_group, mask_idx)
            if subject_key not in subject_group_losses:
                subject_group_losses[subject_key] = []
                
            subject_group_losses[subject_key].append(symm_kl)


    num_subject_groups = 0
    for (subject_group, mask_idx), losses in subject_group_losses.items():
        if losses:
            group_loss = sum(losses)
            kl_losses.append(group_loss)
            num_subject_groups += 1

    kl_loss_term = None
    if kl_losses:
        if use_max:
            kl_loss_term = max(kl_losses)
        else:
            kl_loss_term = sum(kl_losses) / max(num_subject_groups, 1)

    return kl_loss_term, box_masks
