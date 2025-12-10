from setup import _extract_name

def test_simple_name():
    assert _extract_name("My name is Lilly") == "Lilly"

def test_german_pattern():
    assert _extract_name("Ich heiße Max") == "Max"

def test_noise():
    assert _extract_name("My name is... Lilly!!!") == "Lilly"
