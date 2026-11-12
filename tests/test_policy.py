from underwriter import metrics, policy


def test_sections_are_parsed_with_numbers_and_titles():
    s = policy.sections()
    assert list(s)[:3] == ["1.1", "1.2", "1.3"]
    assert len(s) == 22
    assert s["2.5"]["title"] == "Debt-to-income ratios"
    assert not any(line.startswith("#") for line in s["1.5"]["text"].splitlines())


def test_get_section_returns_one_section_or_a_chapter():
    assert policy.get_section("2.5").startswith("[Policy 2.5 Debt-to-income ratios]")
    chapter = policy.get_section("3")
    assert all(f"[Policy 3.{n} " in chapter for n in (1, 2, 3, 4)) and "[Policy 4.1" not in chapter
    assert "No policy section '9.9'" in policy.get_section("9.9")


def test_search_finds_the_right_section_and_cites_it():
    assert policy.search("bankruptcy discharge waiting period").startswith("[Policy 1.2")
    assert "[Policy 3.3 Large deposits]" in policy.search("large deposit source documents")
    assert "[Policy 4.5 Condominiums]" in policy.search("condominium project review")
    assert "No policy section matches" in policy.search("zzzz")


def test_policy_text_states_every_threshold_the_code_uses():
    """Policy text and code cannot drift apart."""
    for section_id, numbers in metrics.THRESHOLDS_BY_SECTION.items():
        text = policy.sections()[section_id]["text"]
        for number in numbers:
            assert str(number) in text or f"{number:,}" in text, (section_id, number)
