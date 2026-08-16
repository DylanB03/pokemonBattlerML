from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import lz4.frame
import torch
from safetensors.torch import save_file as save_safetensors

from pokemon_battler.interaction_modeling import InteractionPolicyHead
from pokemon_battler.large_offline_pipeline import (
    _estimated_interaction_cache_bytes,
    build_parser,
)
from pokemon_battler.metamon_assets import download_selfplay
from pokemon_battler.parallel_interaction_cache import build_parallel_interaction_caches
from pokemon_battler.parallel_trajectory_prepare import (
    prepare_trajectory_dataset_parallel,
)
from pokemon_battler.prepare import SplitConfig
from pokemon_battler.structured_modeling import (
    STRUCTURED_POLICY_SCHEMA,
    initialize_structured_head,
    load_structured_head,
    save_structured_head,
)
from pokemon_battler.structured_train import (
    ShardedInteractionDataset,
    StructuredPolicyCollator,
    train_structured_policy,
)
from pokemon_battler.team_manifest import build_team_manifests
from pokemon_battler.training_data import ShardedJsonlDataset
from pokemon_battler.trajectory_prepare import _trajectory_is_selected
from tests.helpers import state, terminal_state


class LargeOfflineTests(unittest.TestCase):
    def test_pipeline_exposes_four_way_cpu_and_battle_parallelism(self) -> None:
        args = build_parser().parse_args(["--output-dir", "outputs/test-large"])
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.concurrent_games, 4)
        self.assertEqual(args.trajectory_sample_rate, 0.005)
        self.assertEqual(args.maximum_prepared_gib, 32.0)
        self.assertEqual(args.maximum_cache_gib, 16.0)
        self.assertEqual(args.maximum_unmapped_fraction, 0.01)
        self.assertEqual(args.batch_size, 128)
        self.assertGreater(_estimated_interaction_cache_bytes(1), 9_000)

    def test_parallel_preparation_cache_and_resume_preserve_action_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "gen9ou.tar"
            with tarfile.open(archive, "w") as stream:
                for index in range(40):
                    payload = json.dumps(
                        {
                            "states": [state(), terminal_state()],
                            "actions": [0, -1],
                        }
                    ).encode()
                    info = tarfile.TarInfo(
                        "gen9ou/"
                        f"battle-{index}_1800_a_vs_b_01-02-2025_"
                        f"{'WIN' if index % 2 else 'LOSS'}.json"
                    )
                    info.size = len(payload)
                    stream.addfile(info, io.BytesIO(payload))
            prepared = root / "prepared"
            settings = {
                "split_config": SplitConfig(seed=42, train_fraction=0.6, validation_fraction=0.2),
                "workers": 4,
                "shard_trajectories": 5,
                "progress_every": 0,
            }
            with redirect_stdout(io.StringIO()):
                report = prepare_trajectory_dataset_parallel([archive], prepared, **settings)
                resumed = prepare_trajectory_dataset_parallel([archive], prepared, **settings)
            self.assertEqual(report["summary"]["source_items"], 40)
            self.assertEqual(report["summary"]["action_parity"]["unmapped_actions"], 0)
            self.assertEqual(resumed["status"], "complete")
            train = ShardedJsonlDataset(prepared / "train")
            self.assertGreater(len(train), 0)
            self.assertTrue((prepared / "train" / "part-000000.jsonl.offsets.npy").is_file())

            with redirect_stdout(io.StringIO()):
                cache_report = build_parallel_interaction_caches(
                    prepared, root / "cache", workers=4, progress_every=0
                )
            self.assertEqual(cache_report["rows"], 40)
            validation = ShardedInteractionDataset(prepared, root / "cache", "validation")
            batch = StructuredPolicyCollator()([validation[0]])
            self.assertEqual(batch["action_ids"].tolist(), [0])
            self.assertEqual(tuple(batch["legal_action_mask"].shape), (1, 13))

            source = root / "source-checkpoint"
            source.mkdir()
            metadata = {
                "qwen_mode": "lora",
                "interaction_d_model": 32,
                "interaction_attention_heads": 4,
                "interaction_layers": 1,
                "interaction_feedforward_size": 64,
                "interaction_dropout": 0.0,
                "interaction_identity_embedding_size": 8,
            }
            source_head = InteractionPolicyHead(
                16,
                d_model=32,
                attention_heads=4,
                layers=1,
                feedforward_size=64,
                dropout=0.0,
                identity_embedding_size=8,
                qwen_mode="lora",
            )
            save_safetensors(
                {
                    key: value.detach().contiguous()
                    for key, value in source_head.state_dict().items()
                },
                source / "interaction_head.safetensors",
            )
            (source / "training_config.json").write_text(json.dumps(metadata), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                training = train_structured_policy(
                    source_checkpoint=source,
                    prepared_dir=prepared,
                    cache_dir=root / "cache",
                    output_dir=root / "candidate",
                    epochs=1,
                    batch_size=16,
                    eval_batch_size=16,
                    num_workers=0,
                    device_name="cpu",
                    log_steps=100,
                )
            self.assertGreater(training["updates"], 0)
            candidate_metadata = json.loads(
                (root / "candidate" / "training_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(candidate_metadata["qwen_mode"], "lora")
            self.assertEqual(
                candidate_metadata["structured_policy_schema"],
                STRUCTURED_POLICY_SCHEMA,
            )

    def test_outer_lz4_tar_streams_without_writing_an_uncompressed_tar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_tar = io.BytesIO()
            with tarfile.open(fileobj=raw_tar, mode="w") as stream:
                for index in range(8):
                    payload = json.dumps(
                        {
                            "states": [state(), terminal_state()],
                            "actions": [0, -1],
                        }
                    ).encode()
                    inner_payload = lz4.frame.compress(payload)
                    info = tarfile.TarInfo(
                        "gen9ou/"
                        f"battle-{index}_1800_a_vs_b_01-02-2025_"
                        f"{'WIN' if index % 2 else 'LOSS'}.json.lz4"
                    )
                    info.size = len(inner_payload)
                    stream.addfile(info, io.BytesIO(inner_payload))
            archive = root / "gen9ou.tar.lz4"
            archive.write_bytes(lz4.frame.compress(raw_tar.getvalue()))

            with redirect_stdout(io.StringIO()):
                report = prepare_trajectory_dataset_parallel(
                    [archive],
                    root / "prepared",
                    split_config=SplitConfig(
                        seed=42, train_fraction=0.6, validation_fraction=0.2
                    ),
                    workers=2,
                    shard_trajectories=3,
                    progress_every=0,
                    require_outcome=True,
                )

            self.assertEqual(report["summary"]["source_items"], 8)
            self.assertEqual(sum(report["summary"]["transitions_per_split"].values()), 8)
            self.assertTrue(archive.is_file())
            self.assertFalse((root / "gen9ou.tar").exists())
            self.assertFalse((root / "gen9ou.tar.partial").exists())

            limited = root / "prepared-with-limit"
            with (
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "exceeded its storage limit"),
            ):
                prepare_trajectory_dataset_parallel(
                    [archive],
                    limited,
                    split_config=SplitConfig(
                        seed=42, train_fraction=0.6, validation_fraction=0.2
                    ),
                    workers=2,
                    shard_trajectories=3,
                    progress_every=0,
                    require_outcome=True,
                    maximum_output_bytes=1,
                )
            limited_manifest = json.loads(
                (limited / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(limited_manifest["status"], "interrupted")

    def test_sampled_out_stream_member_is_not_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = 0
            while _trajectory_is_selected(
                f"battle-{index}", seed=42, sample_rate=0.05
            ):
                index += 1
            raw_tar = io.BytesIO()
            with tarfile.open(fileobj=raw_tar, mode="w") as stream:
                invalid_payload = b"this is deliberately not trajectory JSON"
                info = tarfile.TarInfo(
                    "gen9ou/"
                    f"battle-{index}_1800_a_vs_b_01-02-2025_WIN.json"
                )
                info.size = len(invalid_payload)
                stream.addfile(info, io.BytesIO(invalid_payload))
            archive = root / "gen9ou.tar.lz4"
            archive.write_bytes(lz4.frame.compress(raw_tar.getvalue()))

            with redirect_stdout(io.StringIO()):
                report = prepare_trajectory_dataset_parallel(
                    [archive],
                    root / "prepared",
                    split_config=SplitConfig(seed=42),
                    trajectory_sample_rate=0.05,
                    workers=1,
                    shard_trajectories=1,
                    progress_every=0,
                    require_outcome=True,
                )

            counters = report["summary"]["counters"]
            self.assertEqual(counters["trajectories_sampled_out"], 1)
            self.assertNotIn("decode_errors", counters)

    def test_selfplay_download_keeps_the_outer_archive_compressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / "download" / "gen9ou.tar.lz4"
            downloaded.parent.mkdir()
            downloaded.write_bytes(b"compressed archive fixture")
            with patch(
                "pokemon_battler.metamon_assets._download_file",
                return_value=downloaded,
            ):
                result = download_selfplay(root, subset="pac-base")
            self.assertEqual(result, downloaded)
            self.assertFalse((root / "self-play" / "pac-base" / "gen9ou.tar").exists())

    def test_structured_sidecar_reuses_interaction_weights_without_disabling_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            metadata = {
                "qwen_mode": "lora",
                "interaction_d_model": 32,
                "interaction_attention_heads": 4,
                "interaction_layers": 1,
                "interaction_feedforward_size": 64,
                "interaction_dropout": 0.0,
                "interaction_identity_embedding_size": 8,
            }
            original = InteractionPolicyHead(
                16,
                d_model=32,
                attention_heads=4,
                layers=1,
                feedforward_size=64,
                dropout=0.0,
                identity_embedding_size=8,
                qwen_mode="lora",
            )
            save_safetensors(
                {key: value.detach().contiguous() for key, value in original.state_dict().items()},
                source / "interaction_head.safetensors",
            )
            sidecar = initialize_structured_head(source, metadata, torch.device("cpu"))
            self.assertEqual(sidecar.qwen_mode, "none")
            self.assertEqual(metadata["qwen_mode"], "lora")
            output = root / "output"
            output.mkdir()
            save_structured_head(sidecar, output)
            (output / "training_config.json").write_text(
                json.dumps(metadata | {"structured_policy_schema": STRUCTURED_POLICY_SCHEMA}),
                encoding="utf-8",
            )
            loaded = load_structured_head(output, torch.device("cpu"))
            self.assertEqual(loaded.qwen_mode, "none")
            self.assertEqual(loaded.d_model, 32)

    def test_team_corpus_splits_by_composition_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teams = root / "teams"
            teams.mkdir()
            for index in range(60):
                (teams / f"team-{index}.txt").write_text(
                    f"Species{index} @ Leftovers\nAbility: Pressure\n- Protect\n",
                    encoding="utf-8",
                )
            report = build_team_manifests(
                [teams], root / "manifests", workers=4, maximum_per_split=None
            )
            self.assertEqual(report["distinct_compositions"], 60)
            self.assertEqual(sum(report["teams_per_split"].values()), 60)
            self.assertTrue((root / "manifests" / "test.txt").is_file())


if __name__ == "__main__":
    unittest.main()
