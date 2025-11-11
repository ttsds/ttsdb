from ttsdb.hfspace import HfSpace

with HfSpace("cdminix/MaskGCT", local=False) as hfspace:
    result = hfspace("This is a test", "test_examples/seven_california_tennis_one_16.wav", "This is Learning English from the News, our podcast about the news headlines.")
    print(result)