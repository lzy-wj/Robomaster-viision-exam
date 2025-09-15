# -*- coding: utf-8 -*-
"""
CIFAR-10 训练脚本 - 简洁稳定版本
"""

import os
# 优化多GPU通信，必须在torch导入之前设置
# 禁用P2P通信，因为混合GPU（如 A6000 和 3090）通常不支持
os.environ['NCCL_P2P_DISABLE'] = '1'
# 禁用InfiniBand，通常在单机多卡或以太网环境中不是必需的，有时会引起初始化问题
os.environ['NCCL_IB_DISABLE'] = '1'

import argparse
import os
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallCifarNet, PowerCifarNet


def auto_detect_hardware():
    """自动检测硬件配置并返回优化参数"""
    config = {
        'device': 'cpu',
        'num_gpus': 0,
        'total_memory_gb': 0,
        'recommended_batch_size': 128,
        'recommended_workers': 4,
        'use_amp': False,
        'use_dataparallel': False
    }
    
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        config['num_gpus'] = torch.cuda.device_count()
        
        # 获取第一个GPU的内存信息
        props = torch.cuda.get_device_properties(0)
        config['total_memory_gb'] = props.total_memory / (1024**3)
        
        print(f"[硬件检测] GPU数量: {config['num_gpus']}")
        for i in range(config['num_gpus']):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"[硬件检测] GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 计算总显存（所有GPU）
        total_memory_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(config['num_gpus'])) / (1024**3)
        print(f"[硬件检测] 总显存: {total_memory_gb:.1f}GB")
        
        # 根据总显存和GPU数量调整批次大小（更保守的策略）
        if total_memory_gb >= 60:  # 60GB+总显存 (3x RTX 3090)
            config['recommended_batch_size'] = 512  # 保守一些
            config['recommended_workers'] = 16
        elif total_memory_gb >= 40:  # 40-60GB总显存
            config['recommended_batch_size'] = 384
            config['recommended_workers'] = 12
        elif total_memory_gb >= 20:  # 20-40GB总显存
            config['recommended_batch_size'] = 256
            config['recommended_workers'] = 8
        else:  # <20GB总显存
            config['recommended_batch_size'] = 128
            config['recommended_workers'] = 4
        
        # 启用混合精度和多GPU
        config['use_amp'] = True
        config['use_dataparallel'] = config['num_gpus'] > 1
        
        print(f"[硬件检测] 推荐批次大小: {config['recommended_batch_size']}")
        print(f"[硬件检测] 推荐worker数: {config['recommended_workers']}")
    else:
        print("[硬件检测] 未检测到GPU，使用CPU训练")
    
    return config


def find_optimal_batch_size(model, device, start_batch_size=128, max_batch_size=4096):
    """自动寻找最大可用批次大小"""
    model.train()
    batch_size = start_batch_size
    
    print(f"[批次优化] 开始寻找最优批次大小，起始: {batch_size}")
    
    while batch_size <= max_batch_size:
        try:
            # 创建测试数据
            test_input = torch.randn(batch_size, 3, 32, 32).to(device)
            test_target = torch.randint(0, 10, (batch_size,)).to(device)
            
            # 清空缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 测试前向传播
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                output = model(test_input)
                loss = torch.nn.functional.cross_entropy(output, test_target)
                loss.backward()
            
            print(f"[批次优化] 批次大小 {batch_size} 可行")
            
            # 尝试更大的批次
            if batch_size < max_batch_size:
                batch_size = min(batch_size * 2, max_batch_size)
            else:
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # 内存不足，使用上一个成功的批次大小
                batch_size = batch_size // 2
                print(f"[批次优化] 内存不足，最终批次大小: {batch_size}")
                break
            else:
                raise e
        except Exception as e:
            print(f"[批次优化] 出现错误: {e}")
            batch_size = batch_size // 2
            break
    
    # 清理测试数据
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return max(batch_size, 32)  # 最小批次大小为32


def get_dataloaders(data_dir, batch_size, num_workers=4):
    """创建数据加载器"""
    # CIFAR-10 标准化参数
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    # 适度数据增强 - 平衡性能和稳定性
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # 只保留轻量级增强，避免过度正则化
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # 减少擦除强度
    ])
    
    # 测试时不增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 检查本地数据是否存在
    data_path = Path(data_dir)
    cifar_path = data_path / 'cifar-10-batches-py'
    download = not cifar_path.exists()
    
    if not download:
        print(f"[信息] 发现本地数据集: {cifar_path}")
    else:
        print(f"[信息] 开始下载CIFAR-10数据集到: {data_path}")
    
    # 创建数据集
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )
    
    # 创建数据加载器 - 高性能配置
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # 保持worker进程
        prefetch_factor=4 if num_workers > 0 else 2,  # 预取数据
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else 2,
    )
    
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, test_loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return correct / total


def train_model(args):
    """主训练函数 - 优化GPU选择和启动流程"""
    print("--- [1/10] 训练启动 ---")
    
    # 自动硬件检测
    hw_config = auto_detect_hardware()
    device = torch.device(hw_config['device'])
    print(f"--- [2/10] 硬件检测完成, 默认主设备: {device} ---")
    
    # 用户参数处理...
    if not hasattr(args, 'auto_optimized') or args.auto_optimized:
        if args.batch_size == 128:
            args.batch_size = hw_config['recommended_batch_size']
        if args.num_workers == 4:
            args.num_workers = hw_config['recommended_workers']
    
    # GPU优化设置
    if hw_config['device'] == 'cuda':
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型
    # ... (模型选择逻辑与之前相同)
    if hw_config['num_gpus'] >= 3 and hw_config['total_memory_gb'] >= 60:
        model = PowerCifarNet(num_classes=10, width_mult=2.5)
        print("[模型选择] 使用PowerCifarNet (大模型)")
    elif hw_config['num_gpus'] >= 2 or hw_config['total_memory_gb'] >= 40:
        model = PowerCifarNet(num_classes=10, width_mult=1.5)
        print("[模型选择] 使用PowerCifarNet (中等模型)")
    else:
        model = SmallCifarNet(num_classes=10)
        print("[模型选择] 使用SmallCifarNet (轻量模型)")
    
    # 先不移动模型到GPU，等待确定最终设备
    print(f"--- [3/10] 模型 '{type(model).__name__}' 在CPU上创建完成 ---")

    # 智能多GPU并行 - 自动选择同种最多的GPU组合
    if hw_config['use_dataparallel']:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(hw_config['num_gpus'])]
        from collections import Counter
        gpu_counts = Counter(gpu_names)
        
        if gpu_counts:
            most_common_gpu = gpu_counts.most_common(1)[0][0]
            device_ids = [i for i, name in enumerate(gpu_names) if name == most_common_gpu]
            
            if len(device_ids) > 1:
                print(f"[GPU优化] 自动选择组合: {most_common_gpu} (共 {len(device_ids)} 张)")
                
                # 核心修复：设置主设备为卡组的第一个，并将模型移到该卡
                device = torch.device(f"cuda:{device_ids[0]}")
                model.to(device)
                print(f"--- [4/10] 主设备变更为 {device}, 模型已移至该GPU ---")
                
                model = nn.DataParallel(model, device_ids=device_ids)
                print(f"--- [5/10] DataParallel 模型包装完成, 使用设备: {device_ids} ---")
            else:
                print(f"[GPU优化] 未找到多张同种GPU，使用默认单GPU {device} 训练。")
                model.to(device) # 移动到默认设备
        else:
            print(f"[GPU优化] 未找到GPU，使用CPU训练。") # CPU场景
    else:
        model.to(device) # 单GPU场景
        print(f"--- [4/10] 单GPU模式, 模型已移至 {device} ---")

    # 移除自动批次查找功能以提高启动速度和稳定性
    print("[优化] 已移除自动批-大小查找功能，请手动指定batch-size。")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"--- [6/10] 模型参数量: {param_count:,} ---")

    print("--- [7/10] 准备加载数据 ---")
    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    print("--- [8/10] Dataloaders 创建完成 ---")

    # ... (优化器和调度器逻辑与之前相同)
    # 损失函数 ...
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # 学习率缩放 ...
    base_lr = args.lr
    # ...
    scaled_lr = base_lr * 2.0 # simplified
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    # 调度器 ...
    if args.batch_size >= 384:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=scaled_lr * 1.5, epochs=args.epochs, steps_per_epoch=len(train_loader))
        scheduler_type = 'OneCycleLR'
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        scheduler_type = 'CosineAnnealingLR'

    print("--- [9/10] 优化器和调度器设置完成 ---")

    # 恢复完整的训练循环
    print("--- [10/10] 开始进入训练循环... ---")
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 初始化AMP scaler（如果使用混合精度）
    scaler = torch.amp.GradScaler('cuda') if hw_config['use_amp'] else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 关键：确保数据发送到正确的主设备
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if hw_config['use_amp'] and scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler_type == 'OneCycleLR':
                scheduler.step()

            running_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            print_freq = max(10, len(train_loader) // 20)
            if batch_idx % print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        test_acc = evaluate(model, test_loader, device)

        if scheduler_type != 'OneCycleLR':
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        gpu_memory = ""
        if torch.cuda.is_available():
            # DataParallel 会将显存分散，我们只关心主设备的
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            gpu_memory = f" | GPU-{device.index}: {memory_used:.1f}/{memory_total:.1f}GB"

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | LR: {current_lr:.6f}{gpu_memory}")

        if test_acc > best_acc:
            best_acc = test_acc
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }, out_dir / 'best_model.pth')
            print(f"  [保存] 新的最佳模型，准确率: {best_acc:.4f}")
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    print(f"[完成] 训练结束，最佳准确率: {best_acc:.4f}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc,
        'final_epoch': args.epochs,
        'config': {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'model': type(model).__name__ if not isinstance(model, nn.DataParallel) else type(model.module).__name__,
            'hardware': hw_config
        }
    }
    
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='训练损失', alpha=0.8)
        plt.title('训练损失变化')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='训练准确率', alpha=0.8)
        plt.plot(test_accs, label='测试准确率', alpha=0.8)
        plt.title('准确率变化')
        plt.xlabel('轮数')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习率曲线
        plt.subplot(1, 3, 3)
        # 重新计算学习率变化（简化版）
        lr_history = []
        temp_optimizer = optim.SGD([torch.tensor(0., requires_grad=True)], lr=args.lr)
        if scheduler_type == 'OneCycleLR':
            temp_scheduler = optim.lr_scheduler.OneCycleLR(temp_optimizer, max_lr=args.lr * 2.0, epochs=args.epochs, steps_per_epoch=len(train_loader))
            for ep in range(args.epochs):
                for _ in range(len(train_loader)):
                    lr_history.append(temp_optimizer.param_groups[0]['lr'])
                    temp_scheduler.step()
        else:
            temp_scheduler = optim.lr_scheduler.CosineAnnealingLR(temp_optimizer, T_max=args.epochs, eta_min=args.min_lr)
            for ep in range(args.epochs):
                lr_history.append(temp_optimizer.param_groups[0]['lr'])
                temp_scheduler.step()
        
        if scheduler_type == 'OneCycleLR':
            plt.plot(lr_history, alpha=0.8)
            plt.title('学习率变化（批次级）')
            plt.xlabel('批次')
        else:
            plt.plot(lr_history, alpha=0.8)
            plt.title('学习率变化（轮次级）')
            plt.xlabel('轮数')
        plt.ylabel('学习率')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[信息] 训练曲线已保存: {out_dir / 'training_curves.png'}")
        
    except ImportError:
        print("[提示] 未安装matplotlib，跳过绘图")
    except Exception as e:
        print(f"[警告] 绘图时出现错误: {e}")
    
    print(f"[信息] 训练历史已保存: {out_dir / 'training_history.json'}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN训练 - 智能优化版')
    
    # 数据和输出
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据集目录')
    parser.add_argument('--out-dir', type=str, default='./runs/cifar10_optimized',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                   help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='批次大小（会根据硬件自动调整）')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='基础学习率（会根据批次大小自动缩放）')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='最小学习率')
    
    # 算力优化参数
    parser.add_argument('--auto-batch-size', action='store_true', default=True,
                        help='自动寻找最优批次大小')
    parser.add_argument('--auto-optimized', action='store_true', default=True,
                        help='启用硬件自动优化')
    parser.add_argument('--disable-auto', action='store_true',
                        help='禁用所有自动优化')
    parser.add_argument('--stable-mode', action='store_true',
                        help='稳定模式：保守的参数设置，优先稳定性')
    
    # 系统参数
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载进程数（会根据硬件自动调整）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 处理特殊模式
    if args.disable_auto:
        args.auto_batch_size = False
        args.auto_optimized = False
        print("[警告] 已禁用所有自动优化，使用用户指定参数")
    
    if args.stable_mode:
        # 稳定模式：保守设置
        args.batch_size = min(args.batch_size, 256)  # 限制批次大小
        args.lr = min(args.lr, 0.1)  # 限制学习率
        args.auto_batch_size = False  # 禁用自动批次搜索
        print("[稳定模式] 使用保守参数设置，优先训练稳定性")
    
    # 设置随机种子
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        import numpy as np
        import random
        np.random.seed(args.seed)
        random.seed(args.seed)
        # 注意：启用benchmark会影响可复现性，但会提升性能
        if not args.auto_optimized:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    print(f"🚀 开始CIFAR-10训练 - 智能优化模式")
    print(f"📊 配置概览: Epochs={args.epochs}, 初始BatchSize={args.batch_size}, LR={args.lr}")
    
    # 开始训练
    train_model(args)


if __name__ == '__main__':
    main()
