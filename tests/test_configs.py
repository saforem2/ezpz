"""Tests for the ezpz.configs module."""

import os
import tempfile
from pathlib import Path

import pytest

try:
    import ezpz.configs as configs
except ImportError:
    configs = None


@pytest.mark.skipif(configs is None, reason="ezpz.configs not available")
class TestConfigs:
    def test_paths_exist(self):
        """Test that important paths exist."""
        assert configs.HERE.exists()
        assert configs.PROJECT_DIR.exists()
        assert configs.CONF_DIR.exists()
        assert configs.BIN_DIR.exists()

    def test_scheduler_detection(self):
        """Test scheduler detection."""
        scheduler = configs.get_scheduler()
        assert scheduler is not None
        assert isinstance(scheduler, str)

    def test_command_exists(self):
        """Test command_exists function."""
        # Test with a command that should exist
        assert configs.command_exists("python") is True
        # Test with a command that should not exist
        assert configs.command_exists("nonexistent_command_xyz") is False

    def test_logging_config(self):
        """Test get_logging_config function."""
        config = configs.get_logging_config()
        assert isinstance(config, dict)
        assert "version" in config
        assert "handlers" in config
        assert "loggers" in config

    def test_train_config(self):
        """Test TrainConfig dataclass."""
        config = configs.TrainConfig(
            model_name_or_path="test-model",
            output_dir="/tmp/test",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="steps",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            num_train_epochs=3.0,
            max_steps=-1,
            lr_scheduler_type="linear",
            warmup_ratio=0.0,
            warmup_steps=0,
            logging_dir="/tmp/test_logs",
            logging_strategy="steps",
            logging_first_step=False,
            logging_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            no_cuda=False,
            seed=42,
            fp16=False,
            fp16_opt_level="O1",
            fp16_backend="auto",
            fp16_full_eval=False,
            local_rank=-1,
            tpu_num_cores=None,
            tpu_metrics_debug=False,
            debug=False,
            dataloader_drop_last=False,
            eval_steps=500,
            dataloader_num_workers=0,
            past_index=-1,
            run_name=None,
            disable_tqdm=False,
            remove_unused_columns=True,
            label_names=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=None,
            ignore_data_skip=False,
            sharded_ddp=False,
            deepspeed=None,
            label_smoothing_factor=0.0,
            adafactor=False,
            group_by_length=False,
            length_column_name="length",
            report_to=["all"],
            dataloader_pin_memory=True,
            skip_memory_metrics=True,
            use_legacy_prediction_loop=False,
            push_to_hub=False,
            resume_from_checkpoint=None,
            hub_model_id=None,
            hub_strategy="every_save",
            hub_token=None,
            hub_private_repo=False,
            gradient_checkpointing=False,
            include_inputs_for_metrics=False,
            fp16_padding=False,
            mp_parameters="",
            auto_find_batch_size=False,
            full_determinism=False,
            torchdynamo=None,
            ray_scope="last",
            ddp_timeout=1800,
            torch_compile=False,
            torch_compile_backend=None,
            torch_compile_mode=None,
        )
        assert config.model_name_or_path == "test-model"
        assert config.output_dir == "/tmp/test"
