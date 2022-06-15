from lm_eval import tasks, evaluator

results = evaluator.simple_evaluate(
    model="gpt2",
    model_args="pretrained=EleutherAI/gpt-neo-125M",
    tasks=["lambada", "hellaswag"],
)
print(evaluator.make_table(results))
