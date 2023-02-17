import random
from collections import Counter

from ml.kv_cache import kv_buffer_scope

fewshot_orig = """The following demonstrates how to choose a good title for a work of fiction, based on a verbal description of the story.

The description is: "Tell me a story about Admiral Craymen eating ice cream."
Given this description, an appropriate title would be: "The Admiral eats ice cream"

The description is: "hey frank! i love your writing. could you tell me a bedtime story?"
Given this description, an appropriate title would be: "Counting sheep"

The description is: "Could you write a story about my OC Monica? She's a catgirl with red hair and a bad attitude"
Given this description, an appropriate title would be: "A red-haired catgirl loses her patience"

The description is: "tell me a story with a dark, creepy atmosphere"
Given this description, an appropriate title would be: "Descent into oblivion"

The description is: "Write a story about a sword-loving ginger butch lesbian who is an indentured servant to a dwindling death cult on Pluto."
Given this description, an appropriate title would be: "A ginger butch lesbian on Pluto practices the blade as she dreams of escaping the death cult"

The description is: "Can you tell me a story about the Borg named Hugh?"
Given this description, an appropriate title would be: "Hugh the Borg"

The description is: "Tell me a story Frank. A story of things you know and don't know. Imagine a story for me Frank. A brand new story to join the night sky."
Given this description, an appropriate title would be: "Known and unknown"

The description is: "Tell me a story about anything you want."
Given this description, an appropriate title would be: ""

The description is: "Tell me a story about the time you got excommunicated"
Given this description, an appropriate title would be: "An excommunication"

The description is: "Can you tell me a story about a robot eating leaves?"
Given this description, an appropriate title would be: "A robot eats leaves"

The description is: "frank, if youve ever watched the animation pop team epic, could you write a fanfic of it?"
Given this description, an appropriate title would be: "Pop Team Epic coffeeshop AU"

The description is: "Write a story about robot cat pirates in a talking airship, please."
Given this description, an appropriate title would be: "Robot cat pirates in a talking airship"

The description is: "Hi Frank. Write me a story about Jim Kirk meeting Jim Morrison, please."
Given this description, an appropriate title would be: "Captain Kirk meets Jim Morrison\""""

_lines = fewshot_orig.split("\n\n")

fewshot_prefix = _lines[0]
fewshot_shots = tuple(lines[1:])


request_format = """The description is: "{text}"
Given this description, an appropriate title would be:"""


def make_shuffled(request):
    shots = list(fewshot_shots)
    random.shuffle(shots)
    return "\n\n".join([fewshot_prefix, *shots, request])


def make_fewshot_titling_request(text):
    return request_format.format(text=text)


def run_fewshot_titling_single_prompt(pr, generator_model, top_p=0.9, temperature=1.0, max_length=30, eos_token_id=198, n=1):
    enc = generator_model.tokenizer

    batch_pr = [pr for _ in range(1)]
    input_ids = enc(
        batch_pr,
    )["input_ids"]

    input_ids_th = torch.as_tensor(input_ids).to(generator_model.device)

    outs = []

    with kv_buffer_scope(generator_model.transformers_model, False):
        for _ in range(n):
            out = generator_model.transformers_model.generate(
                        eos_token_id=eos_token_id,
                        input_ids=input_ids_th,
                        do_sample=True,
                        use_cache=True,
                        top_p=top_p,
                        temperature=temperature,
                        max_length=max_length + input_ids_th.shape[1],
            )
            this = enc.decode(out.cpu().numpy()[0, input_ids_th.shape[1]:])
            this = this[2:].split('"')[0]
            outs.append(this)
    return outs


def run_fewshot_titling(text, generator_model, n_shuffles=8, n_per_shuffle=1, verbose=True, **kwargs):
    request = make_fewshot_titling_request(text)
    counts = Counter()

    for _ in range(n_shuffles):
        prompt = make_shuffled(request)
        if verbose:
            print(repr(prompt))
        outs = run_fewshot_titling_single_prompt(prompt, generator_model, n=n_per_shuffle, **kwargs)
        if verbose:
            print(outs)
        counts.update(outs)

    mc = counts.most_common()
    if verbose:
        print(mc)

    top_count = mc[0][1]
    candidates = [e[0] for e in mc if e[1] == top_count]

    best = sorted(candidates, key=len)[-1]

    return best
