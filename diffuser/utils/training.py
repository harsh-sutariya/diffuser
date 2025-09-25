import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

# Optional wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[ utils/training ] WandB not available - using console logging only")

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        use_wandb=True,
        wandb_project='diffuser',
        wandb_detailed_logging=True,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        
        # Initialize WandB logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_detailed_logging = wandb_detailed_logging
        if self.use_wandb:
            # Extract environment name from results folder path
            env_name = results_folder.split('/')[-3] if '/' in results_folder else 'unknown'
            
            # Get detailed model configuration
            model_config = {
                'batch_size': train_batch_size,
                'learning_rate': train_lr,
                'ema_decay': ema_decay,
                'gradient_accumulate_every': gradient_accumulate_every,
                'step_start_ema': step_start_ema,
                'update_ema_every': update_ema_every,
                'log_freq': log_freq,
                'sample_freq': sample_freq,
                'save_freq': save_freq,
                'environment': env_name,
                
                # Model architecture details
                'horizon': getattr(diffusion_model, 'horizon', 'unknown'),
                'n_timesteps': getattr(diffusion_model, 'n_timesteps', 'unknown'),
                'action_dim': getattr(diffusion_model, 'action_dim', 'unknown'),
                'observation_dim': getattr(diffusion_model, 'observation_dim', 'unknown'),
                'transition_dim': getattr(diffusion_model, 'transition_dim', 'unknown'),
                'action_weight': getattr(diffusion_model, 'action_weight', 'unknown'),
                'loss_type': getattr(diffusion_model, 'loss_type', 'unknown'),
                'predict_epsilon': getattr(diffusion_model, 'predict_epsilon', 'unknown'),
                
                # Dataset info
                'dataset_size': len(dataset),
                'max_path_length': getattr(dataset, 'max_path_length', 'unknown'),
                'normalizer': str(type(getattr(dataset, 'normalizer', 'unknown')).__name__),
            }
            
            # Count model parameters
            total_params = sum(p.numel() for p in diffusion_model.parameters())
            trainable_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
            model_config.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            })
            
            wandb.init(
                project=wandb_project,
                name=f"{env_name}_{os.path.basename(results_folder)}",
                config=model_config
            )
            print(f"[ utils/training ] WandB logging enabled: {wandb.run.name}")
            print(f"[ utils/training ] Logging {trainable_params:,} trainable parameters")
        else:
            print("[ utils/training ] WandB logging disabled")

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            # Track batch statistics for logging
            batch_stats = {'observations': [], 'actions': []}
            
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                
                # Collect batch statistics (only on first accumulation step to avoid overhead)
                if i == 0 and self.use_wandb and self.wandb_detailed_logging and self.step % self.log_freq == 0:
                    with torch.no_grad():
                        if hasattr(batch, 'trajectories'):
                            traj = batch.trajectories
                            obs_dim = self.model.observation_dim if hasattr(self.model, 'observation_dim') else traj.shape[-1] // 2
                            observations = traj[:, :, -obs_dim:]  # Last obs_dim dimensions
                            actions = traj[:, :, :-obs_dim]       # First action_dim dimensions
                            
                            batch_stats['observations'] = {
                                'mean': observations.mean().item(),
                                'std': observations.std().item(),
                                'min': observations.min().item(),
                                'max': observations.max().item(),
                            }
                            batch_stats['actions'] = {
                                'mean': actions.mean().item(),
                                'std': actions.std().item(),
                                'min': actions.min().item(),
                                'max': actions.max().item(),
                            }

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)
                
                # Log checkpoint save event
                if self.use_wandb:
                    wandb.log({
                        'events/checkpoint_saved': 1,
                        'events/checkpoint_epoch': label,
                    }, step=self.step)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                
                # Log to WandB with comprehensive metrics
                if self.use_wandb:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/step': self.step,
                        'train/time_per_step': timer(),
                        'train/epoch': self.step / 10000,  # Assuming 10k steps per epoch
                    }
                    
                    # Add all info metrics from model
                    for key, val in infos.items():
                        if isinstance(val, torch.Tensor):
                            log_dict[f'train/{key}'] = val.item()
                        else:
                            log_dict[f'train/{key}'] = val
                    
                    # Detailed gradient and system statistics (optional for performance)
                    if self.wandb_detailed_logging:
                        # Gradient statistics
                        grad_norm = 0.0
                        param_norm = 0.0
                        grad_max = 0.0
                        grad_min = float('inf')
                        layer_grad_norms = {}
                        
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                param_norm += param.data.norm(2).item() ** 2
                                grad_norm += param.grad.data.norm(2).item() ** 2
                                grad_max = max(grad_max, param.grad.data.abs().max().item())
                                grad_min = min(grad_min, param.grad.data.abs().min().item())
                                
                                # Track layer-wise gradients for key layers
                                layer_grad = param.grad.data.norm(2).item()
                                # Focus on key layers (avoid overwhelming logs)
                                if any(key in name for key in ['down', 'up', 'mid', 'time_embed', 'final']):
                                    layer_name = name.split('.')[0] + '.' + name.split('.')[1] if '.' in name else name
                                    if layer_name not in layer_grad_norms:
                                        layer_grad_norms[layer_name] = []
                                    layer_grad_norms[layer_name].append(layer_grad)
                        
                        grad_norm = grad_norm ** 0.5
                        param_norm = param_norm ** 0.5
                        
                        log_dict.update({
                            'gradients/grad_norm': grad_norm,
                            'gradients/param_norm': param_norm,
                            'gradients/grad_max': grad_max,
                            'gradients/grad_min': grad_min if grad_min != float('inf') else 0.0,
                            'gradients/grad_param_ratio': grad_norm / (param_norm + 1e-8),
                        })
                        
                        # Add layer-wise gradient norms (aggregate multiple params per layer)
                        for layer_name, grad_list in layer_grad_norms.items():
                            if len(grad_list) > 0:
                                total_layer_grad = sum(g**2 for g in grad_list) ** 0.5
                                log_dict[f'gradients/layer_{layer_name}'] = total_layer_grad
                    
                    # Learning rate tracking (current LR from optimizer)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    log_dict['train/learning_rate'] = current_lr
                    
                    if self.wandb_detailed_logging:
                        # Memory usage (if CUDA available)
                        if torch.cuda.is_available():
                            log_dict.update({
                                'system/gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                                'system/gpu_memory_cached_mb': torch.cuda.memory_reserved() / (1024**2),
                            })
                        
                        # EMA model comparison (if EMA is active)
                        if self.step >= self.step_start_ema:
                            ema_param_norm = sum(p.data.norm(2).item() ** 2 for p in self.ema_model.parameters()) ** 0.5
                            param_diff_norm = sum((p1.data - p2.data).norm(2).item() ** 2 
                                                for p1, p2 in zip(self.model.parameters(), self.ema_model.parameters())) ** 0.5
                            log_dict.update({
                                'ema/ema_param_norm': ema_param_norm,
                                'ema/param_diff_norm': param_diff_norm,
                                'ema/diff_ratio': param_diff_norm / (param_norm + 1e-8),
                            })
                        
                        # Batch statistics
                        if batch_stats['observations']:
                            for key, val in batch_stats['observations'].items():
                                log_dict[f'data/observations_{key}'] = val
                        if batch_stats['actions']:
                            for key, val in batch_stats['actions'].items():
                                log_dict[f'data/actions_{key}'] = val
                    
                    wandb.log(log_dict, step=self.step)

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)
                if self.use_wandb:
                    wandb.log({'events/reference_rendered': 1}, step=self.step)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
                if self.use_wandb:
                    wandb.log({'events/samples_generated': 1}, step=self.step)

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
