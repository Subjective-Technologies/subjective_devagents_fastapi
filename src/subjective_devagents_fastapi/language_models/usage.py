"""Provider-reported usage only. No credentials, prompts or speculative token estimates."""
import hashlib
import re


def report(data, trace, model):
    reported_model = data.get('model')
    result = {'model': reported_model if isinstance(reported_model, str) and re.fullmatch(r'[A-Za-z0-9_./:-]{1,128}', reported_model) else model}
    match = re.fullmatch(r'\d{14}-([A-Za-z0-9_-]{1,128})-(?:[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}|[A-Za-z0-9]{8})', trace or '')
    if match:
        result['account'] = hashlib.sha256(match[1].encode()).hexdigest()[:24]
    raw = data.get('usage')
    if not isinstance(raw, dict):
        return result
    def count(value):
        return value if type(value) is int and 0 <= value <= 10**10 else None
    incoming, outgoing = count(raw.get('prompt_tokens')), count(raw.get('completion_tokens'))
    if incoming is None or outgoing is None:
        return result
    result.update(input_tokens=incoming, output_tokens=outgoing)
    details = raw.get('prompt_tokens_details') or {}
    if not isinstance(details, dict): details = {}
    read, write = count(details.get('cached_tokens', 0)), count(details.get('cached_creation_tokens', 0))
    if read is not None and write is not None and read + write <= incoming:
        result.update(cache_read_tokens=read, cache_write_tokens=write)
    total = count(raw.get('total_tokens'))
    result['total_tokens'] = total if total is not None and total >= incoming + outgoing else incoming + outgoing
    return result


class ActivityScope:
    """Provider tool IDs are message-local; channel activity IDs are request-scoped."""
    def __init__(self):
        self.request_id = None
        self.position = 0

    def observe(self, usage):
        if usage.get('status') == 'running' and isinstance(usage.get('id'), str):
            self.request_id = usage['id']
            self.position = 0

    def tool_id(self, provider_id):
        if not self.request_id:
            return provider_id
        identifier = self.request_id + '-' + str(self.position)
        self.position += 1
        return identifier
