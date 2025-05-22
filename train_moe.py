from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch import nn
import wandb

run = wandb.init(
    project="moe-training",  
)

# 自定义MoE层（参考HuggingFace MoE实现范式）
class MoELayer(nn.Module):
    def __init__(self, hidden_size, expert_num=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size*4),
                nn.SiLU(),
                nn.Linear(hidden_size*4, hidden_size)
            ) for _ in range(expert_num)
        ])
        self.gate = nn.Linear(hidden_size, expert_num)
        self.top_k = top_k

    def forward(self, x):
        # 门控计算
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        
        # 专家计算
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = indices[..., i]
            expert_mask = torch.nn.functional.one_hot(expert_idx, num_classes=len(self.experts)).float()
            expert_output = sum(
                expert_mask[..., j].unsqueeze(-1) * self.experts[j](x) 
                for j in range(len(self.experts))
            )
            output += expert_output * weights[..., i].unsqueeze(-1)
        return output

# 改造Qwen2.5模型架构
class QwenMoE(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        print(self.model)
        # 将原FFN层替换为MoE层（示例替换第6-8层）
        for layer in self.model.model.layers:
            layer.mlp = MoELayer(
                hidden_size=base_model.config.hidden_size,
                expert_num=10,
                top_k=2
            )
    def forward(self,**kwargs):
        return self.model(**kwargs)

# 加载预训练参数
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = QwenMoE(base_model)
print('finish model loading')

# 训练参数配置（参考Qwen2.5技术报告的超参数设置[6](@ref)）
training_args = TrainingArguments(
    output_dir="/aifs4su/hansirui_2nd/haoran/moe_save/moe_train_v1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=100,
    fp16=True,
    remove_unused_columns=False,
  
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="steps",
    save_steps=-1,
    report_to="wandb",  # 启用自动日志上报
    run_name=f"moe-7b",  # 唯一运行标识
)
#   gradient_checkpointing=True,

# 示例数据集加载（需替换为实际数据）
from datasets import load_dataset
# dataset = load_dataset("stanfordnlp/imdb")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1") 

# 自定义数据预处理
# def preprocess(examples):
#     # return tokenizer(examples["text"], truncation=True, max_length=2048)
#     return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=2048, return_tensors='pt')

def preprocess(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )
    # 直接将 input_ids 作为 labels
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])

# 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)
trainer.train()