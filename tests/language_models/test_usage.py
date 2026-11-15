import hashlib
from subjective_devagents_fastapi.language_models.usage import report

def test_canonical_usage_and_private_account_reference():
    result=report({'model':'model-x','usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150,
       'prompt_tokens_details':{'cached_tokens':20,'cached_creation_tokens':10}}},'20261115120000-auth-index-1234abcd','fallback')
    assert result=={'model':'model-x','account':hashlib.sha256(b'auth-index').hexdigest()[:24],
       'input_tokens':100,'output_tokens':50,'cache_read_tokens':20,'cache_write_tokens':10,'total_tokens':150}
    assert 'auth-index' not in str(result)

def test_unknown_usage_is_not_zero_and_bad_cache_is_not_guessed():
    assert report({},None,'fallback')=={'model':'fallback'}
    result=report({'usage':{'prompt_tokens':10,'completion_tokens':5,'prompt_tokens_details':{'cached_tokens':100}}},None,'m')
    assert result['total_tokens']==15 and 'cache_read_tokens' not in result


def test_cliproxy_uuidv7_trace_ids_keep_same_account_across_requests():
    first=report({},'20261115120000-0123456789abcdef-019abbcd-1234-7abc-8def-0123456789ab','m')
    second=report({},'20261115120100-0123456789abcdef-019abbcd-5678-7abc-8def-9876543210ab','m')
    assert first['account']==second['account']==hashlib.sha256(b'0123456789abcdef').hexdigest()[:24]
