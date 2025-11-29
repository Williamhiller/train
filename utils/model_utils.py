import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb
from typing import Dict, Any, Optional, Tuple


def load_pretrained_model(
    model_name: str,
    quantization: bool = True,
    quant_bits: int = 4,
    device: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载预训练的Llama模型和对应的分词器
    
    Args:
        model_name: 模型名称或路径
        quantization: 是否使用量化（节省内存）
        quant_bits: 量化位数（4或8）
        device: 设备类型（'cuda', 'cpu'等）
        
    Returns:
        (模型, 分词器) 元组
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在加载模型: {model_name}...")
    print(f"设备: {device}")
    print(f"量化: {'开启' if quantization else '关闭'} ({quant_bits}位)")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",  # Llama模型需要右侧填充
        use_fast=False  # 某些模型可能需要使用非快速分词器
    )
    
    # 添加pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device if device == "cpu" else "auto",
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32
    }
    
    # 使用量化（如果启用）
    if quantization and device == "cuda":
        model_kwargs.update({
            "load_in_4bit": quant_bits == 4,
            "load_in_8bit": quant_bits == 8,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        })
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试以CPU模式加载模型...")
        # 尝试以CPU模式加载
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        device = "cpu"
    
    print(f"模型加载完成！")
    return model, tokenizer


def setup_peft_model(
    model: AutoModelForCausalLM,
    config: Dict[str, Any]
) -> AutoModelForCausalLM:
    """
    使用PEFT（Parameter-Efficient Fine-Tuning）设置模型
    
    Args:
        model: 预训练模型
        config: PEFT配置参数
        
    Returns:
        配置好的PEFT模型
    """
    if not config.get('use_peft', True):
        print("未启用PEFT，将进行全参数微调")
        return model
    
    peft_type = config.get('peft_type', 'lora')
    print(f"设置PEFT模型: {peft_type}")
    
    # 配置LoRA
    if peft_type == 'lora':
        lora_config = LoraConfig(
            r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            target_modules=config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=config.get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        peft_model = get_peft_model(model, lora_config)
        
        # 打印可训练参数数量
        trainable_params = get_trainable_params(peft_model)
        print(f"PEFT模型设置完成")
        print(f"可训练参数: {trainable_params['trainable']:,} / {trainable_params['total']:,} ({trainable_params['percentage']:.2f}%)")
        
        return peft_model
    else:
        print(f"不支持的PEFT类型: {peft_type}，将进行全参数微调")
        return model


def get_trainable_params(model: torch.nn.Module) -> Dict[str, Any]:
    """
    计算模型的可训练参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        包含参数信息的字典
    """
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        num_params = param.numel()
        # 不计算被冻结的参数
        if param.requires_grad:
            trainable_params += num_params
        all_param += num_params
    
    return {
        'trainable': trainable_params,
        'total': all_param,
        'percentage': 100 * trainable_params / all_param
    }


def save_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_path: str,
    is_peft: bool = False
) -> None:
    """
    保存模型和分词器
    
    Args:
        model: 模型
        tokenizer: 分词器
        save_path: 保存路径
        is_peft: 是否为PEFT模型
    """
    print(f"正在保存模型到: {save_path}")
    
    # 创建保存目录
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 保存分词器
    tokenizer.save_pretrained(save_path)
    
    # 保存模型
    if is_peft:
        # 对于PEFT模型，只保存适配器权重
        model.save_pretrained(save_path)
    else:
        # 对于完整模型，保存全部权重
        model.save_pretrained(
            save_path,
            safe_serialization=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    print("模型保存完成！")


def load_fine_tuned_model(
    model_path: str,
    base_model_name: Optional[str] = None,
    quantization: bool = True,
    device: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载微调后的模型
    
    Args:
        model_path: 模型路径
        base_model_name: 基础模型名称（如果是PEFT模型）
        quantization: 是否使用量化
        device: 设备类型
        
    Returns:
        (模型, 分词器) 元组
    """
    import os
    from peft import PeftModel
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查是否为PEFT模型
    is_peft_model = os.path.exists(os.path.join(model_path, 'adapter_config.json'))
    
    if is_peft_model and base_model_name:
        print(f"加载PEFT模型: {model_path}")
        # 先加载基础模型
        base_model, _ = load_pretrained_model(
            base_model_name,
            quantization=quantization,
            device=device
        )
        # 然后加载适配器
        model = PeftModel.from_pretrained(base_model, model_path)
        # 合并权重（可选）
        # model = model.merge_and_unload()
    else:
        print(f"加载完整模型: {model_path}")
        # 直接加载完整模型
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device if device == "cpu" else "auto",
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32
        }
        
        if quantization and device == "cuda":
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16
            })
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    print("微调模型加载完成！")
    return model, tokenizer


def get_model_generation_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从配置中提取模型生成参数
    
    Args:
        config: 配置字典
        
    Returns:
        生成参数字典
    """
    return {
        "max_new_tokens": config.get("max_new_tokens", 100),
        "temperature": config.get("temperature", 0.7),
        "top_p": config.get("top_p", 0.9),
        "do_sample": config.get("do_sample", True),
        "pad_token_id": config.get("pad_token_id"),
        "eos_token_id": config.get("eos_token_id")
    }


def format_training_prompt(input_text: str, output_text: str) -> str:
    """
    格式化训练提示文本
    
    Args:
        input_text: 输入文本
        output_text: 输出文本
        
    Returns:
        格式化的训练文本
    """
    # Llama模型的提示格式（根据实际模型要求调整）
    prompt = f"""{input_text}

{output_text}<|endoftext|>"""
    return prompt.strip()