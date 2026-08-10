import torch
import torch.nn as nn

class SnapKVEvictor:
    """
    SnapKV: Автоматическое вытеснение 95% фонового контекста.
    Сохраняет только Heavy Hitters (H) + Окно Наблюдения (W).
    """
    def __init__(self, max_capacity=2048, window_size=128):
        self.max_capacity = max_capacity
        self.window_size = window_size

    def compress_kv_pair(self, key_states, value_states, attn_weights=None):
        """
        key_states:   [batch, num_heads, seq_len, head_dim]
        value_states: [batch, num_heads, seq_len, head_dim]
        """
        seq_len = key_states.shape[2]
        if seq_len <= self.max_capacity:
            return key_states, value_states

        device = key_states.device
        num_heads = key_states.shape[1]
        obs_start = seq_len - self.window_size

        # 1. Поиск опорных токенов (Heavy Hitters)
        if attn_weights is not None:
            attn_in_obs = attn_weights[:, :, -self.window_size:, :obs_start]
            scores = attn_in_obs.sum(dim=-2)
        else:
            # Резервный выбор по L2-нормам векторов ключей K
            scores = key_states[:, :, :obs_start, :].norm(dim=-1)

        # Выбираем H лучших опорных индексов
        budget_h = self.max_capacity - self.window_size
        topk_indices = torch.topk(scores, k=budget_h, dim=-1).indices
        topk_indices = torch.sort(topk_indices, dim=-1).values

        # 2. Объединяем индексы: (Heavy Hitters) + (Observation Window)
        obs_indices = torch.arange(obs_start, seq_len, device=device).expand(key_states.shape[0], num_heads, -1)
        selected_indices = torch.cat([topk_indices, obs_indices], dim=-1)

        # 3. Извлекаем сжатые тензоры K и V
        selected_exp = selected_indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
        compressed_k = torch.gather(key_states, dim=2, index=selected_exp)
        compressed_v = torch.gather(value_states, dim=2, index=selected_exp)

        return compressed_k, compressed_v


class KIVIKVCache(nn.Module):
    """
    KIVI: Асимметричное INT4 квантование KV-Кэша.
    Keys -> Per-Channel INT4
    Values -> Per-Token INT4
    """
    def __init__(self):
        super().__init__()

    def quantize_key(self, k_tensor):
        """K-кэш квантуется ПО КАНАЛАМ (Per-Channel)"""
        shape = k_tensor.shape
        k_reshaped = k_tensor.reshape(-1, shape[-1])
        absmax = k_reshaped.abs().max(dim=0, keepdim=True)[0]
        scale = absmax / 7.0

        q_signed = torch.clamp(torch.round(k_reshaped / (scale + 1e-8)), -7, 7).to(torch.int8)
        q_unsigned = (q_signed + 8).to(torch.uint8)

        q_flat = q_unsigned.view(-1)
        if q_flat.numel() % 2 != 0:
            q_flat = nn.functional.pad(q_flat, (0, 1))
        q_pairs = q_flat.view(-1, 2)
        packed_k = (q_pairs[:, 0] & 0x0F) | ((q_pairs[:, 1] & 0x0F) << 4)

        return packed_k, scale.half(), shape

    def quantize_value(self, v_tensor):
        """V-кэш квантуется ПО ТОКЕНАМ (Per-Token)"""
        shape = v_tensor.shape
        v_reshaped = v_tensor.reshape(-1, shape[-1])
        absmax = v_reshaped.abs().max(dim=-1, keepdim=True)[0]
        scale = absmax / 7.0

        q_signed = torch.clamp(torch.round(v_reshaped / (scale + 1e-8)), -7, 7).to(torch.int8)
        q_unsigned = (q_signed + 8).to(torch.uint8)

        q_flat = q_unsigned.view(-1)
        if q_flat.numel() % 2 != 0:
            q_flat = nn.functional.pad(q_flat, (0, 1))
        q_pairs = q_flat.view(-1, 2)
        packed_v = (q_pairs[:, 0] & 0x0F) | ((q_pairs[:, 1] & 0x0F) << 4)

        return packed_v, scale.half(), shape

    def dequantize_key(self, packed_k, scale, shape, device, dtype=torch.float16):
        b = packed_k.to(device)
        w0 = (b & 0x0F).to(dtype) - 8.0
        w1 = ((b >> 4) & 0x0F).to(dtype) - 8.0
        q_unpacked = torch.stack([w0, w1], dim=-1).view(-1, shape[-1])
        dequant = q_unpacked * scale.to(device, dtype=dtype)
        return dequant.view(shape)

    def dequantize_value(self, packed_v, scale, shape, device, dtype=torch.float16):
        b = packed_v.to(device)
        w0 = (b & 0x0F).to(dtype) - 8.0
        w1 = ((b >> 4) & 0x0F).to(dtype) - 8.0
        q_unpacked = torch.stack([w0, w1], dim=-1).view(-1, shape[-1])
        dequant = q_unpacked * scale.to(device, dtype=dtype)
        return dequant.view(shape)