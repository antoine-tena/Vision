from network.split_sizes import split_sizes


def test_split_sizes_add_up() -> None:
    sizes = split_sizes()

    assert (sizes.training, sizes.validation, sizes.test) == (59_400, 6_600, 11_000)
    assert sizes.total == 77_000
    assert abs(sum(sizes.share(count) for count in (59_400, 6_600, 11_000)) - 1) < 1e-9
