import torch

from asa.asm_lm import build_model_from_cfg
from asa.checkpoints import load_model_weights, to_canonical
from asa.load_pretrained import load_pretrained

HF_REPO = "DigitalShogun/ASA-ASM-wikitext103-raw"
HF_CKPT = "ASA_ASM_wt103-rawv1_gpt2_T1024_L21_D384_H8_K16_M32_ropek1_alibi1_gamma1_step75000_best.pt"
VARIANTS = ("baseline", "online", "intervene")


def _assert_report_clean(report):
    assert report["disallowed_missing"] == []
    assert report["disallowed_unexpected"] == []
    assert report["disallowed_mismatched"] == []


def test_hf_checkpoint_loads_all_variants():
    for variant in VARIANTS:
        model, report, cfg = load_pretrained(
            repo_id=HF_REPO,
            filename=HF_CKPT,
            variant=variant,
            device="cpu",
        )
        _assert_report_clean(report)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits, _ = model(input_ids)
        assert logits.shape == (1, 8, cfg.vocab_size)
        assert torch.isfinite(logits).all()


def test_roundtrip_across_variants():
    loaded = {}
    for variant in VARIANTS:
        model, report, cfg = load_pretrained(
            repo_id=HF_REPO,
            filename=HF_CKPT,
            variant=variant,
            device="cpu",
        )
        _assert_report_clean(report)
        loaded[variant] = (model, cfg)

    for source_variant, (model, cfg) in loaded.items():
        state_dict = to_canonical(model.state_dict(), "canonical")
        for target_variant in VARIANTS:
            target_model = build_model_from_cfg(cfg, variant=target_variant)
            report = load_model_weights(target_model, state_dict, strict=True)
            _assert_report_clean(report)
