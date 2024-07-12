from steps import model_tester
from utils.test_helper import score


def inference_pipeline(saved_model_name: str) -> None:
    avg_te_correct = model_tester(saved_model_name=saved_model_name)

    print("Your accuracy: {:.02f}%".format(avg_te_correct * 100))
    print("Your score: {:.02f} out of 100".format(score(avg_te_correct * 100)))

    return None
