import translator

def test_translate_english_passthrough(mocker):
    mocker.patch("translator.detect", return_value="en")
    assert translator.translate_to_english("Hello") == "Hello"

def test_translate_non_english_calls_google(mocker):
    mocker.patch("translator.detect", return_value="de")
    mock_gt = mocker.patch("translator.GoogleTranslator")
    mock_gt.return_value.translate.return_value = "Hello world"
    assert translator.translate_to_english("Guten Tag") == "Hello world"
