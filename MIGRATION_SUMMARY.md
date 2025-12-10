# Tóm Tắt Migration: SAM Project từ Flax 0.2.2 sang Flax Linen

## 📋 Tổng Quan

Project SAM đã được migrate từ Flax 0.2.2 (functional API) sang Flax Linen (object-oriented API) với các thư viện mới nhất.

---

## 🔄 Các Thay Đổi Chính

### 1. **Flax API Changes**

#### Code Cũ (Flax 0.2.2):
```python
from flax import nn

class MyModel(nn.Module):
    def apply(self, x, train=True):
        x = nn.Conv(x, features=64, kernel_size=(3, 3))
        x = nn.BatchNorm(x, use_running_average=not train)
        return x

# Khởi tạo
module = MyModel.partial(features=64)
with flax.nn.stateful() as init_state:
    with flax.nn.stochastic(rng):
        _, params = module.init_by_shape(rng, [(shape, dtype)])
        model = flax.nn.Model(module, params)
```

#### Code Mới (Flax Linen):
```python
from flax import linen as nn

class MyModel(nn.Module):
    features: int = 64
    
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x

# Khởi tạo
model = MyModel(features=64)
variables = model.init(rng, dummy_input, train=False)
params = variables['params']
batch_stats = variables.get('batch_stats', {})
```

**Thay đổi chính:**
- `apply()` → `__call__()` với decorator `@nn.compact`
- Functional API (`nn.Conv(x, ...)`) → Object-oriented API (`nn.Conv(...)(x)`)
- `Module.partial()` → Truyền parameters trực tiếp vào constructor
- `flax.nn.Model` và `flax.nn.Collection` → `params` và `batch_stats` dicts
- `flax.nn.stateful()` và `flax.nn.stochastic()` → Không còn cần thiết
- `flax.nn.make_rng()` → `self.make_rng()` trong Module hoặc truyền `rng` trực tiếp

### 2. **Optimizers Changes**

#### Code Cũ:
```python
from flax import optim

optimizer_def = optim.Momentum(learning_rate=0.1, beta=0.9)
optimizer = optimizer_def.create(model)
```

#### Code Mới:
```python
import optax

optimizer = optax.sgd(learning_rate=0.1, momentum=0.9)
opt_state = optimizer.init(params)
```

**Thay đổi:**
- `flax.optim` → `optax`
- API hoàn toàn khác: `optax` sử dụng functional style
- Gradient updates: `updates, new_state = optimizer.update(grads, opt_state, params)`

### 3. **Dependencies Updates**

| Package | Version Cũ | Version Mới |
|---------|-----------|-------------|
| JAX | 0.2.6 | 0.8.1 |
| JAXlib | 0.1.57 | 0.8.1 |
| Flax | 0.2.2 | 0.12.1 |
| TensorFlow | 2.3.1 | 2.20.0 |
| NumPy | 1.18.5 | 2.3.5 |
| Python | 3.7-3.9 | 3.13 |

---

## ✅ Các Phần Đã Hoàn Thành

### 1. **Cấu Trúc Project**
- ✅ Tạo thư mục `sam_new/` cùng cấp với `sam/`
- ✅ Tạo cấu trúc thư mục giống code cũ
- ✅ Tạo các file `__init__.py`

### 2. **Dependencies**
- ✅ Tạo `requirements.txt` với packages mới nhất
- ✅ Tạo venv trong `sam_new/venv/`
- ✅ Cài đặt thành công tất cả dependencies

### 3. **Models - WideResnet**
- ✅ Migrate `wide_resnet.py` sang Flax Linen:
  - `WideResnetBlock` → Object-oriented với `@nn.compact`
  - `WideResnetGroup` → Object-oriented với parameters
  - `WideResnet` → Object-oriented với dataclass fields
  - Loại bỏ dependency vào `FLAGS` trong model definition
- ✅ Migrate `utils.py`:
  - Cập nhật imports sang `flax.linen`
  - Sửa `shake_shake_train` và `shake_drop_train` để nhận `rng` parameter
- ✅ Migrate `load_model.py`:
  - Thay `flax.nn.Model/Collection` → `params` và `batch_stats` dicts
  - Sử dụng `model.init()` thay vì `module.init_by_shape()`
  - Trả về `(model, params, batch_stats)` thay vì `(model, state)`

### 4. **Testing**
- ✅ Tạo `test_model.py` để test model loading
- ✅ Test thành công: Model có thể load và forward pass hoạt động

---

## ❌ Các Phần Còn Thiếu

### 1. **Models Chưa Migrate**

#### `sam_jax/models/pyramidnet.py`
- ❌ Chưa migrate PyramidNet model
- ❌ Chưa migrate PyramidNetShakeDrop
- **Cần làm:**
  - Convert sang Flax Linen API
  - Update imports và dependencies

#### `sam_jax/models/wide_resnet_shakeshake.py`
- ❌ Chưa migrate WideResnetShakeShake
- **Cần làm:**
  - Convert sang Flax Linen
  - Integrate shake-shake regularization với Flax Linen

#### `sam_jax/efficientnet/efficientnet.py`
- ❌ Chưa migrate EfficientNet
- ❌ Chưa migrate các blocks: DepthwiseConv, MBConvBlock, Stem, Head
- **Cần làm:**
  - Convert toàn bộ EfficientNet architecture
  - Update model configs

#### `sam_jax/imagenet_models/resnet.py`
- ❌ Chưa migrate ResNet (ResNet50, ResNet101, ResNet152)
- **Cần làm:**
  - Convert ResNet blocks
  - Update ResNet architecture

#### `sam_jax/imagenet_models/load_model.py`
- ❌ Chưa migrate ImageNet model loading
- ❌ Chưa migrate checkpoint loading cho pretrained models
- **Cần làm:**
  - Update model loading logic
  - Migrate checkpoint utilities

### 2. **Datasets**

#### `sam_jax/datasets/dataset_source.py`
- ❌ Chưa migrate dataset loading cho CIFAR-10/100, SVHN, Fashion-MNIST
- **Cần làm:**
  - Có thể giữ nguyên vì dùng TensorFlow Datasets
  - Cần kiểm tra compatibility

#### `sam_jax/datasets/dataset_source_imagenet.py`
- ❌ Chưa migrate ImageNet dataset loading
- **Cần làm:**
  - Kiểm tra và update nếu cần

#### `sam_jax/datasets/augmentation.py`
- ❌ Chưa migrate augmentation functions
- **Cần làm:**
  - Có thể giữ nguyên vì dùng TensorFlow
  - Cần kiểm tra compatibility với TensorFlow 2.20

### 3. **Training Code**

#### `sam_jax/training_utils/flax_training.py`
- ❌ Chưa migrate training loop
- ❌ Chưa migrate SAM optimizer
- ❌ Chưa migrate learning rate schedules
- ❌ Chưa migrate evaluation code
- **Cần làm:**
  - Convert từ `flax.optim.Optimizer` sang `optax`
  - Implement SAM với Optax
  - Update training step với Flax Linen model API
  - Update evaluation với `model.apply()` và `batch_stats`
  - Migrate checkpoint saving/loading

#### `sam_jax/train.py`
- ❌ Chưa migrate main training script
- **Cần làm:**
  - Update imports
  - Update model initialization
  - Update training function calls

### 4. **Optimizers**

#### `sam_jax/efficientnet/optim.py`
- ❌ Chưa migrate RMSProp optimizer
- ❌ Chưa migrate ExponentialMovingAverage
- **Cần làm:**
  - Convert sang Optax hoặc tạo custom Optax optimizer
  - Update EMA implementation

### 5. **AutoAugment**

#### `autoaugment/autoaugment.py`
- ❌ Chưa migrate AutoAugment
- ❌ Chưa xử lý `tensorflow_addons` dependency (đã deprecated)
- **Cần làm:**
  - Tìm alternative cho `tensorflow_addons.image`
  - Hoặc implement lại các transformations bằng TensorFlow native

#### `autoaugment/policies.py`
- ❌ Có thể giữ nguyên (chỉ là data)

### 6. **Utilities**

#### Checkpointing
- ❌ Chưa migrate checkpoint saving/loading
- **Cần làm:**
  - Sử dụng `orbax-checkpoint` (đã có trong Flax mới)
  - Hoặc `flax.training.checkpoints` nếu còn support

#### Logging & Metrics
- ❌ Chưa migrate TensorBoard logging
- **Cần làm:**
  - Update để tương thích với Flax Linen

---

## 📊 So Sánh File-by-File

| File | Status | Notes |
|------|--------|-------|
| `models/wide_resnet.py` | ✅ Done | Fully migrated to Linen |
| `models/utils.py` | ✅ Done | Updated for Linen |
| `models/load_model.py` | ✅ Done | Updated for Linen |
| `models/pyramidnet.py` | ❌ Missing | Need migration |
| `models/wide_resnet_shakeshake.py` | ❌ Missing | Need migration |
| `efficientnet/efficientnet.py` | ❌ Missing | Need migration |
| `efficientnet/optim.py` | ❌ Missing | Need migration |
| `imagenet_models/resnet.py` | ❌ Missing | Need migration |
| `imagenet_models/load_model.py` | ❌ Missing | Need migration |
| `datasets/dataset_source.py` | ❌ Missing | May work as-is |
| `datasets/dataset_source_imagenet.py` | ❌ Missing | May work as-is |
| `datasets/augmentation.py` | ❌ Missing | May work as-is |
| `training_utils/flax_training.py` | ❌ Missing | Major migration needed |
| `train.py` | ❌ Missing | Need update |
| `autoaugment/autoaugment.py` | ❌ Missing | Need tensorflow_addons alternative |

---

## 🎯 Ưu Tiên Migration

### Priority 1 (Core Functionality):
1. ✅ WideResnet model (DONE)
2. ⏳ Training loop với Optax
3. ⏳ SAM optimizer implementation
4. ⏳ Dataset loading

### Priority 2 (Additional Models):
5. ⏳ PyramidNet
6. ⏳ WideResnetShakeShake
7. ⏳ EfficientNet
8. ⏳ ResNet

### Priority 3 (Utilities):
9. ⏳ Checkpointing
10. ⏳ AutoAugment alternative
11. ⏳ Logging & metrics

---

## 🔧 Các Vấn Đề Đã Gặp và Giải Quyết

### 1. **FLAGS trong Model Definition**
- **Vấn đề:** Không thể dùng `FLAGS` trong model definition trước khi parse
- **Giải pháp:** Truyền flags qua model parameters thay vì dùng global FLAGS

### 2. **PRNG Key Checking**
- **Vấn đề:** `if not prng_key:` không work với JAX arrays
- **Giải pháp:** Dùng `if prng_key is None:`

### 3. **Model Initialization**
- **Vấn đề:** Flax 0.2.2 dùng `init_by_shape()`, Linen dùng `init()`
- **Giải pháp:** Tạo dummy input và dùng `model.init(rng, dummy_input)`

### 4. **Batch Statistics**
- **Vấn đề:** Flax 0.2.2 dùng `Collection`, Linen tách thành `batch_stats`
- **Giải pháp:** Extract `batch_stats` từ `variables` dict

---

## 📝 Notes cho Migration Tiếp Theo

1. **Pattern chung cho Model Migration:**
   - Convert `apply()` → `__call__()` với `@nn.compact`
   - Convert functional calls → object instantiation
   - Loại bỏ `FLAGS` dependencies
   - Update initialization code

2. **Pattern cho Training Migration:**
   - Replace `flax.optim` với `optax`
   - Update gradient computation và application
   - Handle `batch_stats` trong training loop
   - Update checkpoint format

3. **Testing Strategy:**
   - Test từng model riêng lẻ
   - Test training loop với small dataset
   - Verify numerical correctness

---

## 📚 Tài Liệu Tham Khảo

- [Flax Linen Migration Guide](https://flax.readthedocs.io/en/latest/guides/migrate_to_linen.html)
- [Optax Documentation](https://optax.readthedocs.io/)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Last Updated:** 2025-12-10
**Status:** Partial Migration - Core models done, training code pending

