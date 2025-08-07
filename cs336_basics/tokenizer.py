import regex as re 

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
for tok in re.finditer(PAT, "some text that i'll pre-tokenize"):
    print(tok.span(), tok.captures()) # iterate rather storing verything in the mem
