import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import Counter, defaultdict
import heapq

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
compiled_PAT = re.compile(GPT2_PAT)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def chunk_processing(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]
)-> list[tuple[bytes, ...]]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk=f.read(end-start).decode("utf-8", errors="ignore")
    if len(special_tokens)!=0:
        patterns="|".join(re.escape(tok) for tok in special_tokens)
        text=re.split(pattern=patterns, string=chunk)
    else:
        text=[chunk]
    ids: list[tuple[bytes, ...]]=[]
    for txt in text:
        if len(txt)==0:
            continue
        for match in compiled_PAT.finditer(txt):
            token_bytes=match.group(0).encode("utf-8")
            cur_tuple=tuple(bytes([b]) for b in token_bytes)
            ids.append(cur_tuple)
    return ids

class MyPair:
    __slots__=("freq", "pair")

    def __init__(self, freq: int, pair: tuple[bytes, bytes]):
        self.freq=freq
        self.pair=pair

    def __iter__(self):
        yield self.freq
        yield self.pair

    def __lt__(self, other: "MyPair")->bool:
        if self.freq!=other.freq:
            return self.freq>other.freq
        return self.pair>other.pair

    def __eq__(self, value: "MyPair")->bool:
        return self.freq==value.freq and self.pair==value.pair

class RemovableHeap:
    def __init__(self):
        self.q: list[MyPair]=[]
        self.rq: list[MyPair]=[]

    def refresh(self):
        while self.rq and self.rq[0]==self.q[0]:
            heapq.heappop(self.q)
            heapq.heappop(self.rq)

    def push(self, x: MyPair):
        heapq.heappush(self.q, x)

    def remove(self, x: MyPair):
        heapq.heappush(self.rq, x)

    def top(self)->MyPair:
        self.refresh()
        if not self.q:
            raise IndexError("empty queue")
        return self.q[0]

    def pop(self)->MyPair:
        self.refresh()
        if not self.q:
            raise IndexError("empty queue")
        return heapq.heappop(self.q)

    def size(self)->int:
        return len(self.q)-len(self.rq)

    def empty(self)->bool:
        return self.size()==0

    def __len__(self)->int:
        return self.size()

    def __bool__(self)->bool:
        return not self.empty()

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. vocab init
    init_vocab_size=256
    vocab={i: bytes([i]) for i in range(init_vocab_size)}
    for tok in special_tokens:
        vocab[len(vocab)]=tok.encode("utf-8")
    vocab_id=len(vocab)

    # 2. pre-tokenization
    num_processes=4
    with open(input_path, "rb") as f:
        boundaries=find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args=[
        (input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    result_chunk: list[tuple[bytes, ...]]=[]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results=pool.starmap(chunk_processing, chunk_args)
        for chunk_result in results:
            result_chunk.extend(chunk_result)

    tokens_counter: Counter[tuple[bytes, ...]]=Counter()
    pair_counter: Counter[tuple[bytes, bytes]]=Counter()
    pair_dict: defaultdict[tuple[bytes, bytes], set[int]]=defaultdict(set)
    for cur_tuple in result_chunk:
        tokens_counter[cur_tuple]+=1

    words: list[list[bytes]]=[]
    for cur_tuple, freq in tokens_counter.items():
        if freq==0:
            continue
        words.append(list(cur_tuple))

    wid=0
    for cur_tuple, freq in tokens_counter.items():
        if freq==0:
            continue
        for i in range(len(cur_tuple)-1):
            cur_pair=tuple([cur_tuple[i], cur_tuple[i+1]])
            pair_counter[cur_pair]+=freq
            pair_dict[cur_pair].add(wid)
        wid+=1

    pq=RemovableHeap()
    heap_freq: dict[tuple[bytes, bytes], int]={}
    for pair, freq in pair_counter.items():
        if freq==0:
            continue
        pq.push(MyPair(freq, pair))
        heap_freq[pair]=freq

    num_merges=vocab_size-len(vocab)
    merges: list[tuple[bytes, bytes]]=[]

    for i in range(num_merges):
        if pq.empty():
            break

        cur_freq, cur_pair=pq.pop()
        del heap_freq[cur_pair]
        merge_token=cur_pair[0]+cur_pair[1]
        merges.append(cur_pair)
        vocab[vocab_id]=merge_token
        vocab_id+=1

        for word_id in list(pair_dict[cur_pair]):
            old_token=words[word_id]
            old_key=tuple(old_token)
            for j in range(len(old_token)-1):
                old_pair=(old_token[j], old_token[j+1])
                if old_pair in heap_freq:
                    pq.remove(MyPair(pair_counter[old_pair], old_pair))
                    del heap_freq[old_pair]
                pair_counter[old_pair]-=tokens_counter[old_key]
                pair_dict[old_pair].discard(word_id)
                if pair_counter[old_pair]>0:
                    pq.push(MyPair(pair_counter[old_pair], old_pair))
                    heap_freq[old_pair]=pair_counter[old_pair]

            new_token: list[bytes]=[]
            p=0; p0, p1=cur_pair
            tok_len=len(old_token)
            while p+1<tok_len:
                if p<tok_len and old_token[p]==p0 and old_token[p+1]==p1:
                    new_token.append(merge_token)
                    p+=2
                else:
                    new_token.append(old_token[p])
                    p+=1
            if p<tok_len:
                new_token.append(old_token[p])
            words[word_id]=new_token
            new_key=tuple(new_token)
            tokens_counter[new_key]+=tokens_counter[old_key]
            tokens_counter[old_key]=0

            for j in range(len(new_token)-1):
                new_pair=(new_token[j], new_token[j+1])
                if new_pair in heap_freq:
                    pq.remove(MyPair(pair_counter[new_pair], new_pair))
                    del heap_freq[new_pair]
                pair_counter[new_pair]+=tokens_counter[new_key]
                pair_dict[new_pair].add(word_id)
                pq.push(MyPair(pair_counter[new_pair], new_pair))
                heap_freq[new_pair]=pair_counter[new_pair]

    return vocab, merges
    # raise NotImplementedError

if __name__ == "__main__":
    import sys
    import time
    import resource
    import argparse
    import cProfile
    import pstats

    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input-path", required=True, help="Path to training text file")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Max vocabulary size (default: 10000)")
    parser.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"], help="Special tokens (default: ['<|endoftext|>'])")
    parser.add_argument("--num-processes", type=int, default=min(multiprocessing.cpu_count(), 16), help="Parallel pre-tokenization workers")
    parser.add_argument("--profile", action="store_true", help="Profile with cProfile, sorted by tottime")
    parser.add_argument("--top-n", type=int, default=30, help="Number of functions to show in profile output (default: 30)")
    args = parser.parse_args()

    profiler = cProfile.Profile() if args.profile else None
    if profiler: profiler.enable()

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    elapsed = time.perf_counter() - t0

    if profiler: profiler.disable()

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    special_set = {tok.encode() for tok in args.special_tokens}
    longest = max(vocab.values(), key=len)
    regular = [v for v in vocab.values() if v not in special_set]
    longest_reg = max(regular, key=len) if regular else longest

    def _decode(b):
        try: return b.decode()
        except UnicodeDecodeError: return repr(b)

    print(f"\n{'='*60}")
    print(f"Training Done — no files saved")
    print(f"  Time:            {elapsed:.2f}s  ({elapsed/60:.2f} min)")
    print(f"  Peak RSS:        {peak_mb:.1f} MB")
    print(f"  Vocab size:      {len(vocab)}")
    print(f"  Merges:          {len(merges)}")
    print(f"  Longest token:   {_decode(longest)!r}  ({len(longest)} bytes)")
    if longest_reg != longest:
        print(f"  Longest regular: {_decode(longest_reg)!r}  ({len(longest_reg)} bytes)")
    print(f"{'='*60}")

    if profiler:
        print(f"\n{'='*70}")
        print(f"cProfile — top {args.top_n} functions by tottime:")
        print(f"{'='*70}")
        pstats.Stats(profiler).sort_stats("tottime").print_stats(args.top_n)