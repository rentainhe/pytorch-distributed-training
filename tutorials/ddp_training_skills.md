## Pytorch DDP Training Details
### 1. SyncBN
#### 支持多机多卡的 BatchNormalization

- BN包含`moving mean` 和 `moving variance` 两个buffer
- DP模式中的BN被设计为只利用主卡上的结果来计算`moving mean`和`moving variance`，进而广播给其他卡，这样实际上的`batch_size`就只有主卡上`batch_size`大小
当模型很大，batch_size很小的时候，BN操作会限制模型的性能
- `SyncBN` 利用分布式进程通讯接口在各个卡间进行通讯，进而利用所有数据计算BN

### 2. DDP下利用Gradient Accumulation进一步加速
#### 