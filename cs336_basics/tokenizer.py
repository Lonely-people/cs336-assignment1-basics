import os
import pickle
import regex as re
from typing import BinaryIO, Iterable
from cs336_basics.train_bpe import compiled_PAT, Merge_Pair, get_all_pairs

def process_chunk(chunk: str, special_tokens: list[str] | None=None)->list[list[bytes]]:
    special_set=set(special_tokens)
    if len(special_tokens)!=0:
        patterns="|".join(re.escape(tok) for tok in special_tokens)
        patterns=f"({patterns})"
        text=re.split(pattern=patterns, string=chunk)
    else:
        text=[chunk]
    result: list[list[bytes]]=[]
    for txt in text:
        if len(txt)==0:
            continue
        if txt in special_set:
            result.append([txt.encode("utf-8")])
            continue
        for match in compiled_PAT.finditer(txt):
            token_bytes=match.group(0).encode("utf-8")
            cur: list[bytes]=[bytes([b]) for b in token_bytes]
            result.append(cur)
    return result

class Tokenizer:
    def __init__(self, 
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens if special_tokens else []
        known=set(self.vocab.values())
        for tok in self.special_tokens:
            tok_bytes=tok.encode("utf-8")
            if tok_bytes not in known:
                self.vocab[len(self.vocab)]=tok_bytes
                known.add(tok_bytes)
        self.special_tokens=sorted(self.special_tokens or [], key=lambda x: -len(x))
        self.bytes_to_id={v: k for k, v in self.vocab.items()}
        self.merge_ranks={pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None=None):
        vocab: dict[int, bytes]={}
        with open(vocab_filepath, "rb") as f:
            vocab=pickle.load(f)
        merges: list[tuple[bytes, bytes]]=[]
        with open(merges_filepath, "r", encoding="latin-1") as f:
            for line in f:
                line=line.rstrip("\n")
                if not line:
                    continue
                fir, sec=line.split(" ", 1)
                merges.append((fir.encode("latin-1"), sec.encode("latin-1")))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str)->list[int]:
        result=process_chunk(text, self.special_tokens)
        token_ids: list[int]=[]
        for tokens in result:
            tmp=tokens
            while len(tmp)>=2:
                cur_pairs: list[tuple[bytes, bytes]]=get_all_pairs(tmp)
                mn=float('inf'); mn_pair: tuple[bytes, bytes]=()
                for pair in cur_pairs:
                    if pair in self.merge_ranks and mn>self.merge_ranks[pair]:
                        mn=self.merge_ranks[pair]
                        mn_pair=pair
                if mn_pair in self.merge_ranks:
                    tmp=Merge_Pair(tmp, mn_pair)
                else:
                    break
            for token in tmp:
                if token in self.bytes_to_id:
                    token_ids.append(self.bytes_to_id[token])
        return token_ids

    def encode_iterable(self, iterable: Iterable[str])->Iterable[int]:
        for word in iterable:
            id=self.encode(word)
            yield from id

    def decode(self, ids: list[int])->str:
        return b"".join([self.vocab[t] for t in ids]).decode("utf-8", errors="replace")