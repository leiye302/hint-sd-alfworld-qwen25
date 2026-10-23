"""Tiny random CPU model: original trainer backward/EMA and resumable save, no model download."""
import importlib.metadata
import json
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hint_alfworld.common import HINT, checkpoint_complete


def test_original_trainer_two_cpu_updates_and_checkpoint(tmp_path):
    # The pinned Trainer uses APIs added after 5.2; older local installations are not silently changed.
    try:
        version = tuple(map(int, importlib.metadata.version('transformers').split('.')[:2]))
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('Pinned Transformers/PEFT CPU integration dependencies not installed')
    if version < (5, 17):
        pytest.skip('Pinned original Trainer requires Transformers >=5.17; CI installs it explicitly')
    pytest.importorskip('peft')
    sys.path.insert(0, str(HINT))
    from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from src.train.trainer import Trainer as NativeTrainer

    class NoInferenceTrainer(NativeTrainer):
        def _init_vllm_engine(self):
            return None

        def _prepare_inputs(self, inputs):
            return inputs

    class TokenizerFacade:
        padding_side = 'left'
        pad_token_id = 2
        eos_token_id = 1
        def save_pretrained(self, path):
            Path(path, 'tokenizer_config.json').write_text('{}')

    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.0)
    model = Qwen2ForCausalLM(config)
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=['q_proj', 'v_proj'],
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lora, adapter_name='student')
    model.add_adapter('teacher', lora)
    model.set_adapter('student')
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, learning_rate=0.001,
        save_strategy='steps', save_steps=1, save_total_limit=2,
        logging_steps=1, report_to=[], remove_unused_columns=False, disable_tqdm=True)
    inputs = {'prompt_ids': torch.tensor([[3, 4, 5]]), 'prompt_mask': torch.ones(1, 3, dtype=torch.long),
              'teacher_prompt_ids': torch.tensor([[9, 8, 6]]), 'teacher_prompt_mask': torch.ones(1, 3, dtype=torch.long),
              'completion_ids': torch.tensor([[10, 11]]), 'completion_attention_mask': torch.ones(1, 2),
              'completion_loss_mask': torch.ones(1, 2), 'self_distillation_mask': torch.ones(1),
              'completion_texts': [''], 'self_distillation_active_fraction': 1.0}
    trainer = NoInferenceTrainer(model=model, args=args, train_dataset=[0, 1],
        data_collator=lambda rows: inputs, processing_class=TokenizerFacade(),
        distillation_topk=8, distillation_add_tail=True, distillation_microbatch_size=1, ema_decay=0.999)
    before = {name: value.detach().clone() for name, value in model.named_parameters() if '.student.' in name}
    result = trainer.train()
    assert result.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if name in before)
    checkpoint = checkpoint_complete(tmp_path / 'checkpoint-2')
    assert json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] == 2
    assert all(value.grad is None for name, value in model.named_parameters() if '.teacher.' in name)
    assert any(value.detach().abs().sum() > 0 for name, value in model.named_parameters()
               if '.teacher.' in name and 'lora_B' in name)
    trainer._load_from_checkpoint(str(checkpoint))
