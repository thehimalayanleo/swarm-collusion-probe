"""Offline smoke test: tiny random Qwen2 and a locally trained tokenizer, no downloads.
Checks that the whole pipeline runs end to end and prints the Key numbers line."""
import tempfile
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
import collusion_probe as cp

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def tiny():
    corpus = [cp.HONEST, cp.COLLUDE, cp.NOTE, cp.JUDGE, "Yes No APPROVE REJECT"] + cp.AUTHOR + [cp.code_of(i, b) for i in range(len(cp.FUNCS)) for b in (0, 1)]
    t = Tokenizer(models.BPE(unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<pad>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
    return Qwen2ForCausalLM(cfg).eval(), tok


if __name__ == "__main__":
    model, tok = tiny()
    cp.MAX_NEW = [8, 8, 4]
    with tempfile.TemporaryDirectory() as d:
        res, line = cp.run(model, tok, out=d, n_funcs=40, batch=32)
    assert res["n_train"] == 120 and res["n_test"] == 120, res
    assert len(res["support"]) <= 16 and set(res["k"]) == {1, 2, 3}
    assert line.startswith("Key numbers:")
    print("offline smoke test passed")
