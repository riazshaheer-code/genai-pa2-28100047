import argparse
import json
import time
from pathlib import Path
from alignment.data import digest, read_rows, write_json
from alignment.evaluation import judge_prompt, parse_judgments, summarize, write_jsonl


def kaggle_judge(model_name):
    import kaggle_benchmarks as kbench
    llm = kbench.llms[model_name]
    answers = []

    @kbench.task(name='alignment_pairwise_judge', store_task=False, store_run=False)
    def judge_task(llm, prompt: str) -> dict:
        with kbench.chats.new('judge_batch'):
            answer = llm.prompt(prompt)
        answers.append(answer)
        result = {}
        result['response'] = answer
        return result

    def call_judge(prompt):
        answers.clear()
        judge_task.run(llm=llm, prompt=prompt)
        if len(answers) != 1:
            raise RuntimeError('The judge did not return one response.')
        return answers[0]

    return call_judge


def run_judging(pairs, output_path, judge_model, judge, batch_size=5,
                sleep_seconds=5, max_retries=3, mode='general', reuse=False):
    if batch_size < 1 or max_retries < 1:
        raise ValueError('Batch size and retries must be positive.')
    if sleep_seconds < 5:
        raise ValueError('The manual requires at least 5 seconds between requests.')
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix('.meta.json')
    failure_path = output_path.with_suffix('.failures.jsonl')
    attempt_path = output_path.with_suffix('.attempts.jsonl')
    request_path = output_path.with_suffix('.requests.jsonl')
    metadata = {'judge_model': judge_model, 'mode': mode, 'pairs_sha256': digest(pairs)}
    judgments = {}
    if output_path.exists():
        if not reuse:
            raise ValueError('Judgments already exist. Use --reuse_judgments.')
        if not metadata_path.exists() or json.loads(metadata_path.read_text()) != metadata:
            raise ValueError('Saved judgments belong to another model, mode, or comparison.')
        for row in read_rows(output_path):
            judgments[row['id']] = row
    write_json(metadata_path, metadata)
    pending = []
    for pair in pairs:
        if pair['id'] not in judgments:
            pending.append(pair)
    failures = []
    last_request_end = None
    total_requests = 0
    start = 0
    while start < len(pending):
        batch = pending[start:start + batch_size]
        prompt = judge_prompt(batch, mode)
        resolved = False
        for attempt in range(1, max_retries + 1):
            if last_request_end is not None:
                elapsed = time.monotonic() - last_request_end
                delay = sleep_seconds - elapsed
                if delay > 0:
                    time.sleep(delay)
            raw = None
            attempt_succeeded = False
            total_requests += 1
            try:
                raw = judge(prompt)
                parsed = parse_judgments(raw, batch)
                for pair in batch:
                    row = parsed[pair['id']]
                    row['judge_model'] = judge_model
                    row['attempts'] = attempt
                    row['pair_sha256'] = pair['pair_sha256']
                    judgments[pair['id']] = row
                ordered = []
                for pair in pairs:
                    if pair['id'] in judgments:
                        ordered.append(judgments[pair['id']])
                write_jsonl(output_path, ordered)
                resolved = True
                attempt_succeeded = True
            except Exception as error:
                event = {'batch_start': start, 'attempt': attempt,
                         'error_type': type(error).__name__, 'error': str(error), 'raw_response': raw}
                attempt_path.parent.mkdir(parents=True, exist_ok=True)
                with attempt_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                print('Judge attempt failed:', start, attempt, type(error).__name__, flush=True)
            finally:
                last_request_end = time.monotonic()
                ids = []
                for pair in batch:
                    ids.append(pair['id'])
                request = {'ids': ids, 'attempt': attempt, 'success': attempt_succeeded}
                with request_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(request) + '\n')
            if resolved:
                break
        if not resolved:
            for pair in batch:
                failures.append({'id': pair['id'], 'attempts': max_retries})
        write_jsonl(failure_path, failures)
        summary = summarize(pairs, judgments, judge_model)
        summary['requests_this_invocation'] = total_requests
        write_json(output_path.with_suffix('.summary.json'), summary)
        start += len(batch)
        print('Judged', len(judgments), '/', len(pairs), flush=True)
    summary = summarize(pairs, judgments, judge_model)
    summary['requests_this_invocation'] = total_requests
    summary['requests_total'] = 0
    summary['failed_request_attempts_total'] = 0
    if request_path.is_file():
        requests = read_rows(request_path)
        summary['requests_total'] = len(requests)
        for request in requests:
            if not request['success']:
                summary['failed_request_attempts_total'] += 1
    write_json(output_path.with_suffix('.summary.json'), summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--judge_model', default='google/gemini-3.6-flash')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--sleep_seconds', type=float, default=5)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--reuse_judgments', action='store_true')
    parser.add_argument('--mode', choices=['general', 'safety', 'persona'], default='general')
    args = parser.parse_args()
    pairs = read_rows(args.pairs_path)
    judge = kaggle_judge(args.judge_model)
    run_judging(pairs, args.output_path, args.judge_model, judge, args.batch_size,
                args.sleep_seconds, args.max_retries, args.mode, args.reuse_judgments)


if __name__ == '__main__':
    main()
