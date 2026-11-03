"""Bind the Pascal FP8 KV opt-in to the kernels in the actual build."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

OLD_SUPPORT = '''    unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .is_ok_and(|major| major >= 8)
    }
'''
NEW_SUPPORT = '''    let (Ok(major), Ok(minor)) = (
        unsafe {
            result::device::get_attribute(
                device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
        },
        unsafe {
            result::device::get_attribute(
                device,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
        },
    ) else {
        return false;
    };
    fp8_support::supports_arch(
        major,
        minor,
        mistralrs_paged_attn::USE_FP8,
        mistralrs_paged_attn::USE_FP8_LEGACY_SM61,
    )
'''


def install(repo):
    repo = Path(repo)
    path = repo / "mistralrs-core/src/paged_attention/cache_engine.rs"
    if not path.exists():
        return False
    changed = False

    def replace(path, old, new, count=1):
        nonlocal changed
        text = path.read_text()
        if text.count(new) == count and old not in text.replace(new, ""):
            return
        if text.count(old) != count or new in text:
            raise RuntimeError(f"unrecognized paged FP8 contract: {path}")
        path.write_text(text.replace(old, new, count))
        changed = True

    replace(path, OLD_SUPPORT, NEW_SUPPORT)
    marker = '#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]'
    replace(path, marker,
            '#[cfg(all(test, feature = "cuda", target_family = "unix"))]\n'
            '#[test]\nfn campaign_cuda_fp8_kv_sm61_device_gate() {\n'
            '    let device = Device::new_cuda(0).expect("physical SM61 GPU");\n'
            '    assert!(mistralrs_paged_attn::USE_FP8_LEGACY_SM61);\n'
            '    assert!(cuda_supports_fp8(&device));\n}\n\n' + marker)
    replace(path, 'const METAL_PAGED_LAYOUT_ENV:',
            '#[cfg(any(test, all(feature = "cuda", target_family = "unix")))]\n'
            '#[path = "fp8_support.rs"]\nmod fp8_support;\n\nconst METAL_PAGED_LAYOUT_ENV:')
    replace(path, 'FP8 KV cache requires CUDA compute capability 8.0 or newer and a matching CUDA build',
            'FP8 KV cache requires CUDA SM80+ or an explicitly enabled SM61 software FP8 build (cuda-legacy-fp8)')
    build = repo / "mistralrs-paged-attn/build.rs"
    replace(build, '    println!("cargo::rustc-check-cfg=cfg(has_fp8)");',
            '    println!("cargo::rustc-check-cfg=cfg(has_fp8)");\n'
            '    println!("cargo::rustc-check-cfg=cfg(has_fp8_legacy_sm61)");', count=3)
    replace(build, '    if using_fp8 {\n',
            '    if using_fp8 && allow_legacy_fp8 && compute_cap == 61 {\n'
            '        println!("cargo:rustc-cfg=has_fp8_legacy_sm61");\n'
            '    }\n\n    if using_fp8 {\n')
    replace(repo / "mistralrs-paged-attn/src/cuda/mod.rs",
            'pub const USE_FP8: bool = cfg!(has_fp8);',
            'pub const USE_FP8: bool = cfg!(has_fp8);\n'
            'pub const USE_FP8_LEGACY_SM61: bool = cfg!(has_fp8_legacy_sm61);')
    for source, target in (
        ("paged_fp8_support.rs", "mistralrs-core/src/paged_attention/fp8_support.rs"),
        ("campaign_paged_fp8.rs", "mistralrs-core/tests/campaign_paged_fp8.rs"),
        ("legacy_fp8_backend.rs", "mistralrs-core/src/paged_attention/legacy_fp8_backend.rs"),
    ):
        destination = repo / target
        content = (ROOT / source).read_text()
        if not destination.exists() or destination.read_text() != content:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
            changed = True
    install_backend_routing(repo, replace)
    return changed


def install_backend_routing(repo, replace):
    config = repo / "mistralrs-core/src/paged_attention/config.rs"
    replace(config,
            'pub(crate) struct AttentionFeaturesConfig<M> {\n    base: M,\n    fallback: Option<PrefixPrefillAttentionFeatures>,\n}',
            'pub(crate) struct AttentionFeaturesConfig<M> {\n    base: M,\n    fallback: Option<PrefixPrefillAttentionFeatures>,\n    force_standard: bool,\n}')
    replace(config, '        Self { base, fallback }',
            '        Self {\n            base,\n            fallback,\n            force_standard: false,\n        }')
    text = config.read_text()
    start = text.index('impl<M: std::ops::Deref> ModelConfigLike for AttentionFeaturesConfig<M>')
    end = text.index('#[derive(Clone)]\npub struct ModelConfigMetadata', start)
    part = text[start:end]
    replacements = {
        '        self.base.attention_backend_kind()': '        if self.force_standard {\n            AttentionBackendKind::Standard\n        } else {\n            self.base.attention_backend_kind()\n        }',
        '        self.base.attention_backend_kind_for_layer(layer_idx)': '        if self.force_standard {\n            AttentionBackendKind::Standard\n        } else {\n            self.base.attention_backend_kind_for_layer(layer_idx)\n        }',
        '        self.base.kv_cache_layout()': '        if self.force_standard {\n            select_kv_cache_layout(self.base.kv_cache_layout(), self)\n        } else {\n            self.base.kv_cache_layout()\n        }',
        '        self.base.kv_cache_layout_for_layer(layer_idx)': '        if self.force_standard {\n            select_kv_cache_layout_for_layer(self.base.kv_cache_layout_for_layer(layer_idx), self, layer_idx)\n        } else {\n            self.base.kv_cache_layout_for_layer(layer_idx)\n        }',
    }
    already_patched = 'if self.force_standard' in part
    normalize = lambda value: re.sub(r'\s+', '', re.sub(r',\s*\)', ')', value))
    for old, new in replacements.items():
        if already_patched:
            if normalize(new) not in normalize(part):
                raise RuntimeError("modified legacy FP8 backend delegation")
        else:
            if part.count(old) != 1:
                raise RuntimeError("unrecognized legacy FP8 backend delegation")
            part = part.replace(old, new, 1)
    updated = text[:start] + part + text[end:]
    marker = 'include!("legacy_fp8_backend.rs");'
    if marker not in updated:
        updated += '\n' + marker + '\n'
    if updated != text:
        config.write_text(updated)
    replace(repo / "mistralrs-core/src/paged_attention/mod.rs",
            'pub(crate) use config::with_attention_features;',
            'pub(crate) use config::{with_attention_features, with_legacy_fp8_backend};')
    for file, old, indent in (
        ("normal.rs", '    let model_metadata =\n        crate::paged_attention::with_attention_features(model.model_config(), attention_features);', '    '),
        ("multimodal.rs", '        let model_metadata = crate::paged_attention::with_attention_features(\n            model.model_config(),\n            self.inner\n                .activation_model_config(&runtime_config)?\n                .attention_features,\n        );', '        '),
    ):
        new = old + '\n' + indent + 'let model_metadata = crate::paged_attention::with_legacy_fp8_backend(\n' + indent + '    model_metadata,\n' + indent + '    paged_attn_config.as_ref(),\n' + indent + '    &device,\n' + indent + ');'
        replace(repo / "mistralrs-core/src/pipeline" / file, old, new)
