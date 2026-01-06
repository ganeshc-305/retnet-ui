"""
Training engine for RetNet model with real-time metrics streaming.
Supports custom text datasets and generator-based progress reporting.
"""

import os
import math
import time
from typing import Generator, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

from retnet_model import RetNet, RetNetConfig

# Handle AMP compatibility across PyTorch versions
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = None


class LocalTextDataset(Dataset):
    """Dataset for loading and tokenizing local text files."""
    
    def __init__(self, text_content: str, seq_len: int = 512, tokenizer=None):
        """
        Initialize dataset from text content.
        
        Args:
            text_content: Raw text string to tokenize
            seq_len: Sequence length for each sample
            tokenizer: Tokenizer instance (defaults to GPT2)
        """
        self.seq_len = seq_len
        
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text_content, add_special_tokens=False)
        
        # Split into fixed-length sequences
        n_chunks = len(tokens) // seq_len
        if n_chunks == 0:
            raise ValueError(f"Text too short! Need at least {seq_len} tokens, got {len(tokens)}")
        
        tokens = tokens[:n_chunks * seq_len]
        self.data = torch.tensor(tokens, dtype=torch.long).view(-1, seq_len)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'input_ids': self.data[idx], 'targets': self.data[idx]}


class TrainingEngine:
    """Training engine with generator-based progress reporting."""
    
    def __init__(
        self,
        model: RetNet,
        config: RetNetConfig,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 100,
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 4,
        use_amp: bool = True
    ):
        """
        Initialize training engine.
        
        Args:
            model: RetNet model to train
            config: Model configuration
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            gradient_clip: Gradient clipping threshold
            gradient_accumulation_steps: Steps to accumulate gradients
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and device == 'cuda'
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=weight_decay
        )
        
        self.global_step = 0
        self.running_loss = 0.0
        self.running_grad_norm = 0.0
        
        if self.use_amp:
            self.scaler = GradScaler(AMP_DEVICE) if AMP_DEVICE else GradScaler()
        else:
            self.scaler = None
    
    def _get_lr(self, step: int, max_steps: int, base_lr: float) -> float:
        """Calculate learning rate with warmup and cosine decay."""
        if step < self.warmup_steps:
            return base_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (max_steps - self.warmup_steps)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _set_lr(self, lr: float):
        """Set learning rate for optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_step(
        self,
        train_loader: DataLoader,
        max_steps: int,
        eval_interval: int = 100,
        val_loader: Optional[DataLoader] = None,
        save_interval: int = 500,
        checkpoint_dir: str = './checkpoints'
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Training loop that yields metrics at each step.
        
        Args:
            train_loader: Training data loader
            max_steps: Maximum training steps
            eval_interval: Steps between validation
            val_loader: Validation data loader (optional)
            save_interval: Steps between checkpoint saves
            checkpoint_dir: Directory to save checkpoints
        
        Yields:
            Dictionary with current training metrics
        """
        self.model.train()
        train_iter = iter(train_loader)
        base_lr = self.optimizer.param_groups[0]['lr']
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        accumulated_loss = 0.0
        micro_step = 0
        start_time = time.time()
        
        while self.global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass with AMP
            if AMP_DEVICE:
                with autocast(AMP_DEVICE, enabled=self.use_amp):
                    _, loss, _ = self.model(input_ids, mode='parallel', targets=targets)
            else:
                with autocast(enabled=self.use_amp):
                    _, loss, _ = self.model(input_ids, mode='parallel', targets=targets)
            
            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            micro_step += 1
            
            # Optimizer step after accumulation
            if micro_step % self.gradient_accumulation_steps == 0:
                # Update learning rate
                current_lr = self._get_lr(self.global_step, max_steps, base_lr)
                self._set_lr(current_lr)
                
                # Gradient clipping and optimizer step
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    
                    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.running_grad_norm = 0.99 * self.running_grad_norm + 0.01 * grad_norm.item()
                    else:
                        self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        self.optimizer.step()
                        self.running_grad_norm = 0.99 * self.running_grad_norm + 0.01 * grad_norm.item()
                
                self.optimizer.zero_grad()
                
                self.running_loss = 0.9 * self.running_loss + 0.1 * accumulated_loss
                self.global_step += 1
                
                # Prepare metrics
                metrics = {
                    'step': self.global_step,
                    'loss': accumulated_loss,
                    'running_loss': self.running_loss,
                    'lr': current_lr,
                    'grad_norm': self.running_grad_norm,
                    'progress': self.global_step / max_steps,
                    'elapsed_time': time.time() - start_time
                }
                
                if self.device == 'cuda':
                    metrics['memory_gb'] = torch.cuda.memory_allocated() / 1e9
                
                # Validation
                if val_loader is not None and self.global_step % eval_interval == 0:
                    val_loss = self._evaluate(val_loader)
                    metrics['val_loss'] = val_loss
                    metrics['perplexity'] = math.exp(min(val_loss, 20))
                
                # Save checkpoint
                if self.global_step % save_interval == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'checkpoint_step{self.global_step}.pt'
                    )
                    self._save_checkpoint(checkpoint_path)
                    metrics['checkpoint_saved'] = checkpoint_path
                
                accumulated_loss = 0.0
                yield metrics
    
    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader, max_batches: int = 50) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            if AMP_DEVICE:
                with autocast(AMP_DEVICE, enabled=self.use_amp):
                    _, loss, _ = self.model(input_ids, mode='parallel', targets=targets)
            else:
                with autocast(enabled=self.use_amp):
                    _, loss, _ = self.model(input_ids, mode='parallel', targets=targets)
            
            total_loss += loss.item()
            n_batches += 1
            
            if n_batches >= max_batches:
                break
        
        self.model.train()
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }, path)
