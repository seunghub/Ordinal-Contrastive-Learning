def ordinal_infoNCE_loss(x, label):
    x = F.normalize(x, dim=-1)
    x_sim = x@x.T
    label_dist = (label.unsqueeze(1)-label.unsqueeze(0)).abs().float() # Label Distance (e.g., CN-EMCI=1 CN-LMCI=2)
    # label_dist = (label_dist).bool().float()
    tau = 0.1/label_dist; tau[label_dist==0]=0
    
    # REMOVE DIAGONAL ENTRIES (SELF SIMILARITY)
    diag_mask = torch.eye(x_sim.size(0), dtype=torch.bool).to(x.device)
    x_sim = x_sim.masked_select(~diag_mask).view(x_sim.size(0), -1)
    tau = tau.masked_select(~diag_mask).view(tau.size(0), -1)
    inv_tau = 1/tau; inv_tau[tau==0]=0
        
    # HANDLE NEGATIVE PAIRS
    negative_mask = tau.clone().bool().float() # Mark negative pair as 1. positive pair as 0.
    x_sim_div_tau = x_sim * inv_tau
    x_sim_div_tau[torch.isinf(x_sim_div_tau)] = 0
    exp_x_sim_div_tau = torch.exp(x_sim_div_tau)
    exp_x_sim_div_tau = exp_x_sim_div_tau * negative_mask # exp(neg_sim/tau)
    negative_logit = torch.sum(exp_x_sim_div_tau, dim=-1)
    neg_sim = exp_x_sim_div_tau[negative_mask!=0] # For check progress
    exp_x_sim_div_tau_div_tau = exp_x_sim_div_tau * inv_tau 
    exp_x_sim_div_tau_div_tau[exp_x_sim_div_tau_div_tau!=exp_x_sim_div_tau_div_tau] = 0
    exp_x_sim_div_tau_div_tau = exp_x_sim_div_tau_div_tau * negative_mask # exp(neg_sim/tau)/tau
    
    # HANDLE POSITIVE PAIRS
    positive_tau = torch.sum(exp_x_sim_div_tau, dim=-1) / torch.sum(exp_x_sim_div_tau_div_tau, dim=-1)
    positive_mask = torch.ones_like(negative_mask) - negative_mask
    x_sim_div_tau = x_sim / positive_tau.unsqueeze(1)
    exp_x_sim_div_tau = torch.exp(x_sim_div_tau) # exp(pos_sim/pos_tau)
    pos_sim = exp_x_sim_div_tau[positive_mask!=0]
    all_logit = negative_logit + torch.sum(exp_x_sim_div_tau * positive_mask, dim=-1)
    exp_x_sim_div_tau_div_neg_logit = exp_x_sim_div_tau / all_logit.unsqueeze(1) # exp(pos_sim/pos_tau) / sum(exp(neg_sim/tau))
    loss = torch.sum(-torch.log(exp_x_sim_div_tau_div_neg_logit) * positive_mask,dim=1) / torch.sum(positive_mask,dim=1)
    
    return loss
