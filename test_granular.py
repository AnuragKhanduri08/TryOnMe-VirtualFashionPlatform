
import sys
print("Start", flush=True)

try:
    print("Importing tokenizers...", flush=True)
    import tokenizers
    print("tokenizers imported", flush=True)
except Exception as e:
    print(f"tokenizers failed: {e}", flush=True)

try:
    print("Importing huggingface_hub...", flush=True)
    import huggingface_hub
    print("huggingface_hub imported", flush=True)
except Exception as e:
    print(f"huggingface_hub failed: {e}", flush=True)

try:
    print("Importing regex...", flush=True)
    import regex
    print("regex imported", flush=True)
except Exception as e:
    print(f"regex failed: {e}", flush=True)

try:
    print("Importing transformers...", flush=True)
    import transformers
    print("transformers imported", flush=True)
except Exception as e:
    print(f"transformers failed: {e}", flush=True)

try:
    print("Importing CLIPTokenizer...", flush=True)
    from transformers import CLIPTokenizer
    print("CLIPTokenizer imported", flush=True)
except Exception as e:
    print(f"CLIPTokenizer failed: {e}", flush=True)

try:
    print("Importing modeling_utils...", flush=True)
    from transformers import modeling_utils
    print("modeling_utils imported", flush=True)
except Exception as e:
    print(f"modeling_utils failed: {e}", flush=True)

try:
    print("Importing CLIPModel...", flush=True)
    from transformers import CLIPModel
    print("CLIPModel imported", flush=True)
except Exception as e:
    print(f"CLIPModel failed: {e}", flush=True)

print("Done", flush=True)
