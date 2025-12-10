from setup import _parse_password

def test_parse_digits_direct():
    assert _parse_password("1234") == "1234"

def test_parse_digits_spoken_en():
    assert _parse_password("one two three four") == "1234"

def test_parse_digits_spoken_de():
    assert _parse_password("eins zwei drei vier") == "1234"

def test_parse_mixed_speech():
    assert _parse_password("my password is one 2 drei four") == "1234"
