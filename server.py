import os
import re
from typing import List, Optional, Dict, Any, Union

import torch
import deepspeed
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import pycountry
from langdetect import detect as ld_detect
from langdetect.lang_detect_exception import LangDetectException


SIZE_TO_REPO = {
    "600M": "facebook/nllb-200-distilled-600M",
    "1.3B": "facebook/nllb-200-distilled-1.3B",
    "3.3B": "facebook/nllb-200-3.3B",
}


def pick_model_id() -> str:
    override = os.getenv("NLLB_MODEL_ID", "").strip()
    if override:
        return override
    size = os.getenv("NLLB_MODEL_SIZE", "600M").strip()
    if size not in SIZE_TO_REPO:
        raise ValueError(
            f"Unsupported NLLB_MODEL_SIZE={size}. Use one of {list(SIZE_TO_REPO.keys())} or set NLLB_MODEL_ID."
        )
    return SIZE_TO_REPO[size]


def pick_dtype() -> torch.dtype:
    dt = os.getenv("DTYPE", "fp16").strip().lower()
    if dt in ("fp16", "float16"):
        return torch.float16
    if dt in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dt in ("fp32", "float32"):
        return torch.float32
    raise ValueError("DTYPE must be one of: fp16, bf16, fp32")


def pick_tp_size() -> int:
    tp = os.getenv("TP_SIZE", "auto").strip().lower()
    if tp == "auto":
        return max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    return max(1, int(tp))


class TranslateRequest(BaseModel):
    q: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to translate."
    )
    target: str = Field(
        ..., description="Target language (ISO 639-1 or BCP-47 like 'es', 'zh-TW')."
    )
    source: Optional[str] = Field(
        None,
        description="Optional source language (ISO 639-1 or BCP-47). If omitted, best-effort detect.",
    )
    format: Optional[str] = Field(
        "text", description="Compatibility field; 'text' or 'html'."
    )
    model: Optional[str] = Field(None, description="Compatibility field; ignored.")
    max_new_tokens: int = Field(128, ge=1, le=2048)
    num_beams: int = Field(1, ge=1, le=16)
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    do_sample: bool = Field(False)
    truncate_input: bool = Field(True)


app = FastAPI(title="NLLB-200 Inference API")

MODEL_ID = pick_model_id()
DTYPE = pick_dtype()
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1024"))

tokenizer = None
model = None
ds_engine = None

ISO_TO_NLLB: Dict[str, str] = {}
ISO_NAME: Dict[str, str] = {}
BCP47_SPECIALS = {
    "zh-cn": "zho_Hans",
    "zh-sg": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh-hk": "zho_Hant",
    "zh-mo": "zho_Hant",
    "zh-hant": "zho_Hant",
}


def _pycountry_name(iso: str) -> Optional[str]:
    try:
        if len(iso) == 2:
            lang = pycountry.languages.get(alpha_2=iso)
        else:
            lang = pycountry.languages.get(alpha_3=iso)
        return getattr(lang, "name", None) if lang else None
    except Exception:
        return None


def _iso1_from_iso3(iso3: str) -> Optional[str]:
    try:
        lang = pycountry.languages.get(alpha_3=iso3)
        return getattr(lang, "alpha_2", None) if lang else None
    except Exception:
        return None


def _normalize_bcp47(code: str) -> str:
    return code.strip().replace("_", "-").lower()


def _preferred_variant(nllb_codes: List[str]) -> str:
    # Prefer Latn variants first
    for s in ("Latn", "Cyrl", "Arab", "Deva", "Hans", "Hant"):
        for c in nllb_codes:
            if c.endswith("_" + s):
                return c
    return nllb_codes[0]


def _build_language_maps():
    global ISO_TO_NLLB, ISO_NAME

    lang_code_to_id = _get_lang_code_to_id()
    codes = sorted(lang_code_to_id.keys())
    by_iso3: Dict[str, List[str]] = {}
    for c in codes:
        if "_" not in c:
            continue
        iso3, _script = c.split("_", 1)
        by_iso3.setdefault(iso3.lower(), []).append(c)

    ISO_TO_NLLB = {}
    ISO_NAME = {}

    for iso3, variants in by_iso3.items():
        chosen = _preferred_variant(sorted(variants))
        iso1 = _iso1_from_iso3(iso3)

        ISO_TO_NLLB[iso3] = chosen
        ISO_NAME.setdefault(iso3, _pycountry_name(iso3) or iso3)

        if iso1:
            iso1 = iso1.lower()
            ISO_TO_NLLB[iso1] = chosen
            ISO_NAME.setdefault(iso1, _pycountry_name(iso1) or iso1)

    # BCP-47 region/script special-cases
    for k, v in BCP47_SPECIALS.items():
        if v in lang_code_to_id:
            ISO_TO_NLLB[k] = v


def _resolve_to_nllb(code: str) -> str:
    if not code:
        raise HTTPException(status_code=400, detail="Missing language code.")
    norm = _normalize_bcp47(code)

    # Exact match
    if norm in ISO_TO_NLLB:
        return ISO_TO_NLLB[norm]

    # Language-only fallback
    lang_part = norm.split("-", 1)[0]
    if lang_part in ISO_TO_NLLB:
        return ISO_TO_NLLB[lang_part]

    raise HTTPException(status_code=400, detail=f"Unsupported language code: '{code}'")


def _lang_to_id(nllb_tgt: str) -> int:
    lang_code_to_id = _get_lang_code_to_id()
    if nllb_tgt not in lang_code_to_id:
        raise HTTPException(
            status_code=400, detail=f"Unknown NLLB target code '{nllb_tgt}'"
        )
    return int(lang_code_to_id[nllb_tgt])


_NLLB_LANG_RE = re.compile(r"^[a-z]{3}_[A-Za-z]{4}$")


def _get_lang_code_to_id() -> Dict[str, int]:
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id
    vocab = tokenizer.get_vocab()
    return {tok: idx for tok, idx in vocab.items() if _NLLB_LANG_RE.match(tok)}


def _detect_iso639_1(text: str) -> Optional[str]:
    try:
        return ld_detect(text).lower()
    except LangDetectException:
        return None
    except Exception:
        return None


def _ensure_batch(q: Union[str, List[str]]) -> List[str]:
    return [q] if isinstance(q, str) else q


def _detect_one(text: str) -> Dict[str, Any]:
    iso = _detect_iso639_1(text) or "en"
    return {"language": iso}


@app.on_event("startup")
def _startup():
    global tokenizer, model, ds_engine

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.cuda.is_available():
        try:
            deepspeed.init_distributed()
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if torch.cuda.is_available():
        tp_size = pick_tp_size()
        ds_engine = deepspeed.init_inference(
            model=model,
            dtype=DTYPE,
            replace_with_kernel_inject=True,
            tensor_parallel={"tp_size": tp_size},
        )
    else:
        ds_engine = None

    _build_language_maps()


@torch.inference_mode()
def _translate_batch(
    texts: List[str], src_nllb: str, tgt_nllb: str, gen: Dict[str, Any]
) -> List[str]:
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large: {len(texts)} > MAX_BATCH_SIZE={MAX_BATCH_SIZE}",
        )

    forced_bos_token_id = _lang_to_id(tgt_nllb)

    tokenizer.src_lang = src_nllb
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=gen.get("truncate_input", True),
        max_length=MAX_INPUT_LENGTH,
    )
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}

    gen_model = ds_engine.module if ds_engine is not None else model
    generated = gen_model.generate(
        **enc,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=gen.get("max_new_tokens", 128),
        num_beams=gen.get("num_beams", 1),
        do_sample=gen.get("do_sample", False),
        temperature=gen.get("temperature", 1.0),
        top_p=gen.get("top_p", 1.0),
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "dtype": str(DTYPE),
        "max_batch_size": MAX_BATCH_SIZE,
        "max_input_length": MAX_INPUT_LENGTH,
    }


@app.post("/language/translate/v2")
def translate(req: TranslateRequest) -> Dict[str, Any]:
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    texts = _ensure_batch(req.q)
    if not texts:
        raise HTTPException(status_code=400, detail="Missing q")

    tgt_nllb = _resolve_to_nllb(req.target)

    detected_iso: Optional[str] = None
    if req.source:
        src_nllb = _resolve_to_nllb(req.source)
    else:
        detected_iso = _detect_iso639_1(texts[0]) or "en"
        try:
            src_nllb = _resolve_to_nllb(detected_iso)
        except HTTPException:
            detected_iso = "en"
            src_nllb = _resolve_to_nllb("en")

    out = _translate_batch(
        texts,
        src_nllb=src_nllb,
        tgt_nllb=tgt_nllb,
        gen=req.model_dump(),
    )

    translations = []
    for s in out:
        item = {"translatedText": s}
        if req.source is None:
            item["detectedSourceLanguage"] = detected_iso
        translations.append(item)

    return {"data": {"translations": translations}}


@app.get("/language/translate/v2")
def translate_get(
    q: List[str] = Query(
        ..., description="Text to translate. Repeat for batching: ?q=a&q=b"
    ),
    target: str = Query(..., description="Target language (ISO 639-1 or BCP-47)"),
    source: Optional[str] = Query(None, description="Optional source language"),
    format: Optional[str] = Query(
        "text", description="Compatibility field; 'text' or 'html'."
    ),
    model: Optional[str] = Query(None, description="Compatibility field; ignored."),
) -> Dict[str, Any]:
    req = TranslateRequest(
        q=q, target=target, source=source, format=format, model=model
    )
    return translate(req)


class DetectRequest(BaseModel):
    q: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to detect."
    )


@app.post("/language/translate/v2/detect")
def detect(req: DetectRequest) -> Dict[str, Any]:
    texts = _ensure_batch(req.q)
    if not texts:
        raise HTTPException(status_code=400, detail="Missing q")

    detections = [[_detect_one(t)] for t in texts]
    return {"data": {"detections": detections}}


@app.get("/language/translate/v2/detect")
def detect_get(
    q: List[str] = Query(
        ..., description="Text to detect. Repeat for batching: ?q=a&q=b"
    ),
) -> Dict[str, Any]:
    req = DetectRequest(q=q)
    return detect(req)


@app.get("/language/translate/v2/languages")
def languages(
    target: Optional[str] = Query(
        None,
        description="Optional target language for localized names (compatibility).",
    ),
) -> Dict[str, Any]:
    langs = []
    seen = set()

    for iso in ISO_TO_NLLB.keys():
        if "-" in iso:
            continue
        if iso in seen:
            continue
        seen.add(iso)

        entry = {"language": iso}
        if target:
            entry["name"] = ISO_NAME.get(iso, iso)
        langs.append(entry)

    langs.sort(key=lambda x: x["language"])
    return {"data": {"languages": langs}}
