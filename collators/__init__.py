COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls

# NOTE: the training-time collator (.qwen3_vl) is intentionally omitted in this
# inference-only copy; nothing here registers into COLLATORS.
