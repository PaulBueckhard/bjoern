from server import persona

def test_persona_en():
    p = persona("en", "Lilly")
    assert "Björn" in p
    assert "short" in p.lower()
    assert "Lilly" in p
    assert "never talk about adult topics" in p.lower()

def test_persona_de():
    p = persona("de", "Max")
    assert "Björn" in p
    assert "Max" in p
    assert "kindgerecht" in p.lower()
