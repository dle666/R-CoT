import argparse
import json
import re
import os

def extract_choice(q, answer, i):
    pattern0 = r"Answer:([A-D])"
    match = re.search(pattern0, answer)
    if match:
        return match.group(1)
    # Pattern 1: Therefore, the correct answer is option {choice}

    pattern1 = r"the correct answer is option ([A-D])"
    match = re.search(pattern1, answer)
    if match:
        return match.group(1)

    # Pattern 2: Therefore, option {choice} is selected
    pattern2 = r"option ([A-D]) is selected"
    match = re.search(pattern2, answer)
    if match:
        return match.group(1)

    # Pattern 3: Therefore, the answer is option {choice}
    pattern3 = r"Answer: ([A-D])"
    match = re.search(pattern3, answer)
    if match:
        return match.group(1)
    #
    # Pattern 4: Therefore, the answer is (C), The answer is (D) 14.
    pattern4 = r"([A-D])"
    match = re.search(pattern4, answer)
    if match:
        return match.group(1)

    # Pattern 5: find the solution from the last sentence
    sentences = answer.split(".")
    try:
        last_sentence = sentences[-2].strip()
    except:
        return None
    match = re.search(r'is ([A-D])', last_sentence)
    if match:
        answer = match.group(1)
        return answer
    print(last_sentence)

    return None

def calculate_accuracy(predictions_file, ground_truth_file):
    with open(predictions_file, "r") as f_pred, open(ground_truth_file, "r") as f_gt:
        predictions = [json.loads(line) for line in f_pred]
        predictions_ids = [item["question_id"] for item in predictions]
        questions = [item["prompt"] for item in predictions]
        predictions = [item["text"] for item in predictions]
        ground_truth = [json.loads(line) for line in f_gt]
        ground_truth_dict = {}
        for grd in ground_truth:
            ground_truth_dict.update({grd["question_id"] : grd["text"]})
        ground_truth = [ground_truth_dict[i] for i in predictions_ids]
        # Extract choices from predictions
        predicted_choices = [extract_choice(q, a, i) for i, (q, a) in enumerate(zip(questions, predictions))]
        none_num = len([x for x in predicted_choices if x is None])
        print(f"nones : {none_num}")

        # Calculate accuracy
        total = len(ground_truth)
        correct = 0
        correct_list, wrong_list = [], []
        for q, gt, pred_c, pred in zip(questions, ground_truth, predicted_choices, predictions):
            if pred_c == gt:
                correct+=1
                correct_list.append({"question": q, "pred": pred})
            else:
                wrong_list.append({"question": q, "pred": pred, "gt": gt})
        accuracy = correct / total * 100
        if args.save_correct_wrong:
            correct_file, wrong_file = os.path.splitext(predictions_file)[0]+"_correct.json", os.path.splitext(predictions_file)[0]+"_wrong.json"
            with open(correct_file, "w") as f:
                json.dump(correct_list, f)
            with open(wrong_file, "w") as f:
                json.dump(wrong_list, f)
        if args.debug:
            with open(args.question_file, "r") as f_question:
                questions = [json.loads(line) for line in f_question]
            correct_result_indexs = [index for index,(pred, gt) in enumerate(zip(predicted_choices, ground_truth)) if pred == gt]
            correct_result = [(index, questions[index], predictions[index], ground_truth[index]) for index in correct_result_indexs]
            print(0)

        return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str, default="/root/paddlejob/workspace/env_run/dlluo/dle/G-LLaVA-main/playground/data/test_answers.jsonl")
    parser.add_argument("--predictions_file", type=str, default="/root/paddlejob/workspace/env_run/dlluo/G-LLaVA-main-copy/results/answer1.jsonl")
    parser.add_argument("--question_file", type=str, default="/root/paddlejob/workspace/env_run/dlluo/dle/G-LLaVA-main/playground/data/test_questions.jsonl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_correct_wrong", action="store_true")
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.predictions_file, args.ground_truth_file)
    print(f"Accuracy: {accuracy:.2f}%")