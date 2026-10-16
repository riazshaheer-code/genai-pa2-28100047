from alignment.data import choose_rows, extract_sft, extract_uf


def test_sft_uses_first_exchange_from_multiturn_conversation():
    messages = [
        {'role': 'user', 'content': ' First question '},
        {'role': 'assistant', 'content': ' First answer '},
        {'role': 'user', 'content': 'Follow-up requiring earlier history'},
        {'role': 'assistant', 'content': 'Follow-up answer'},
    ]
    assert extract_sft({'messages': messages}) == {
        'prompt': 'First question', 'response': 'First answer'
    }


def test_sft_rejects_missing_messages_and_invalid_first_exchange():
    for messages in [None, [], [{}], 'invalid',
                     [{'role': 'system'}, {'role': 'assistant', 'content': 'answer'}],
                     [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': 'answer'}]]:
        assert extract_sft({'messages': messages}) is None


def test_preference_filter_still_rejects_multiturn_pairs():
    chosen = [
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': 'answer'},
        {'role': 'user', 'content': 'follow-up'},
        {'role': 'assistant', 'content': 'chosen follow-up'},
    ]
    rejected = list(chosen)
    rejected[-1] = {'role': 'assistant', 'content': 'rejected follow-up'}
    assert extract_uf({'chosen': chosen, 'rejected': rejected}) is None


def test_sft_sampling_excludes_validation_first_prompts():
    source = []
    for index in range(5):
        messages = [
            {'role': 'user', 'content': str(index)},
            {'role': 'assistant', 'content': 'answer'},
            {'role': 'user', 'content': 'follow-up'},
            {'role': 'assistant', 'content': 'later answer'},
        ]
        source.append({'messages': messages})
    selected = choose_rows(source, extract_sft, 4, 42, excluded_prompts={'0'})
    assert len(selected) == 4
    assert all(row['prompt'] != '0' for row in selected)
    assert selected == choose_rows(source, extract_sft, 4, 42, excluded_prompts={'0'})
