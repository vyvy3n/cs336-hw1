

encoding_type = "utf-8"
test_string = "hello! こんにちは!"
# test_string = "こ"
utf8_encoded = test_string.encode(encoding_type)
print(utf8_encoded)
# Get the byte values for the encoded string (integers from 0 to 255).
list(utf8_encoded)
# One byte does not necessarily correspond to one Unicode character!
print(len(test_string))
print(len(utf8_encoded))
print(max(utf8_encoded))
print(utf8_encoded.decode(encoding_type))

def decode_utf8_bytes_to_str(bytestring):
    return "".join([bytes([b]).decode('utf-8') for b in bytestring])

print(decode_utf8_bytes_to_str(utf8_encoded))
"""
utf-8 encoding 
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
13
23
hello! こんにちは!

utf-16
b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00S0\x930k0a0o0!\x00'
13
28
hello! こんにちは!

utf-32
b'\xff\xfe\x00\x00h\x00\x00\x00e\x00\x00\x00l\x00\x00\x00l\x00\x00\x00o\x00\x00\x00!\x00\x00\x00 \x00\x00\x00S0\x00\x00\x930\x00\x00k0\x00\x00a0\x00\x00o0\x00\x00!\x00\x00\x00'
13
56
hello! こんにちは!
"""