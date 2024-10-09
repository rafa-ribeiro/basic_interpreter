from basic_interpreter import basic


def test__function_call_works():
    command = "FUN soma(a, b) -> a + b"
    _, _ = basic.run("<stdin>", command)
    result, _ = basic.run("<stdin>", "soma(3, 5)")
    assert result.elements[0].value == 8


def test__function_call_with_too_few_args():
    command = "FUN soma(a, b) -> a + b"
    _, _ = basic.run("<stdin>", command)
    result, error = basic.run("<stdin>", "soma(2)")
    assert error.details == "1 too few args passed into 'soma'"


def test__function_call_with_too_many_args():
    command = "FUN soma(a, b) -> a + b"
    _, _ = basic.run("<stdin>", command)
    result, error = basic.run("<stdin>", "soma(2, 3, 3)")
    assert error.details == "1 too many args passed into 'soma'"
