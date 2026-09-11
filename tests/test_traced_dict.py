from skverify.sets import TracedDict

def test_traced_dict_get_returns_default_for_missing_key():
    d = TracedDict({0.0: "zero", 1.0: "one"})
    result = d.get(5.0, "not found")
    assert result == "not found"

def test_traced_dict_get_returns_value_for_existing_key():
    d = TracedDict({0.0: "zero", 1.0: "one"})
    result = d.get(1.0, "not found")
    assert result == "one"