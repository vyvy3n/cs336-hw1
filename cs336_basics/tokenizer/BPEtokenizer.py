import regex as re
from collections import defaultdict
import multiprocessing as mp
from typing import List, Tuple

class BPE:
    def __init__(self,input_path, vocab_size, special_tokens):
        self.vocab_size = vocab_size 
        self.special_tokens = special_tokens
        self.input_path = input_path
        self.merges = []
    
    def _strip_special_tokens(self, text: str) -> str:
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        pattern = "|".join(escaped_tokens)
        parts = re.split(pattern, text)
        return "".join(parts)
    
    def _chunk_text_at_special_tokens(self, text: str, num_chunks: int) -> List[str]:
        if not self.special_tokens:
            chunk_size = len(text) // num_chunks
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        pattern = "|".join(escaped_tokens)
        boundaries = []
        for match in re.finditer(pattern, text):
            boundaries.extend([match.start(), match.end()])
        
        boundaries = sorted(set([0] + boundaries + [len(text)]))
        target_chunk_size = len(text) // num_chunks
        chunks = []
        current_chunk_start = 0
        
        for i in range(1, len(boundaries)):
            chunk_end = boundaries[i]
            current_chunk_size = chunk_end - current_chunk_start
            if (current_chunk_size >= target_chunk_size or 
                len(chunks) == num_chunks - 1 or 
                i == len(boundaries) - 1):
                
                chunks.append(text[current_chunk_start:chunk_end])
                current_chunk_start = chunk_end
                
                if len(chunks) == num_chunks:
                    break
        if current_chunk_start < len(text):
            if chunks:
                chunks[-1] += text[current_chunk_start:]
            else:
                chunks.append(text[current_chunk_start:])
        
        return chunks
    
    def _process_chunk(self, chunk: str) -> List[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        cleaned_chunk = self._strip_special_tokens(chunk)
        tokens = re.findall(PAT, cleaned_chunk)
        return tokens
    
    def parallel_pretokenize(self, text: str, num_processes: int = None) -> List[str]:
        if num_processes is None:
            num_processes = mp.cpu_count()
        chunks = self._chunk_text_at_special_tokens(text, num_processes)
        with mp.Pool(num_processes) as pool:
            chunk_results = pool.map(self._process_chunk, chunks)
        all_tokens = []
        for chunk_tokens in chunk_results:
            all_tokens.extend(chunk_tokens)
        
        return all_tokens
    
    def train_bpe(self, use_parallel: bool = True, num_processes: int = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(self.input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if use_parallel:
            regexed = self.parallel_pretokenize(text[:10000], num_processes)
        else:
            text = self._strip_special_tokens(text)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            regexed = re.findall(PAT, text[:10000])
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for token in self.special_tokens:
            vocab[len(vocab)] = token.encode('utf-8')
        utfs = []
        for piece in regexed:
            piece_bytes = piece.encode('utf-8')
            utfs.append(list(piece_bytes))  
        next_token_id = len(vocab) 
        
        for i in range(self.vocab_size - len(vocab)):
            counts = self._count_occurences(utfs)
            if not counts:
                break
            chosen_pair = max(counts, key=counts.get)  # tuple of (int, int)
            utfs = self._merge_pair(utfs, chosen_pair, next_token_id)
            
            left_bytes = vocab[chosen_pair[0]]
            right_bytes = vocab[chosen_pair[1]]
            new_token = left_bytes + right_bytes
            vocab[next_token_id] = new_token
            
            self.merges.append((left_bytes, right_bytes))
            next_token_id += 1
            
        return vocab, self.merges
                    
    def _merge_pair(self, utfs, pair_to_merge, new_token_id): 
        new_utfs = []
        for word in utfs:
            new_token = []
            i = 0
            while i < len(word): 
                if i < len(word) - 1 and (word[i], word[i+1]) == pair_to_merge:
                    new_token.append(new_token_id)
                    i += 2
                else:
                    new_token.append(word[i])
                    i += 1
            new_utfs.append(new_token)
        return new_utfs
                
    def _count_occurences(self,utfs):
        counts = defaultdict(int)
        for tok in utfs:                      
            for i in range(len(tok) - 1):
                pair = (tok[i], tok[i + 1]) 
                counts[pair] += 1
        return counts
                    
        
            

def test_train_bpe_special_tokens():
    path = '/home/ed/work/other_projects/stanLLMasingments/StanfP1/data/TinyStoriesV2-GPT4-train.txt'
    tokenizer = BPE(path, 1000, ["<|endoftext|>"])
    vocab, merges = tokenizer.train_bpe()
    print(vocab)
    print("########################")
    print(merges)   



if __name__ == "__main__":
    test_train_bpe_special_tokens()
