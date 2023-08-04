from transformers import GPT2Tokenizer

class Prot2TextTokenizer(GPT2Tokenizer):   
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )
        
         
    # def convert_tokens_to_string(self, tokens):
    #     if tokens is not None:
    #         if len(tokens)>0:
    #             tokens = [i for i in tokens if i is not None]
    #             text = "".join(tokens)
    #             text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
    #             return text
    #         else:
    #             return ''
    #     else:
    #         return ''