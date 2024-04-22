# CS839
Data profiling project

Main task:

>Input: Multi-Columns Tables without column name
>
>Output: Same number of column names 
>
>Result: F1 score

## Experiments

### Baseline

Model: Llama2

Type: 70B-GPTQ

Input: Multiple columns without column name

Output: Same number of column names (If the result number doesn't match, setting the result as mismatch)

Result: F1 score

```json
{"accuracy": 0.2689075630252101, "macro avg": {"precision": 0.1241846398249352, "recall": 0.06902952073505617, "f1-score": 0.07897371190700836, "support": 1190}, "weighted avg": {"precision": 0.48347263814793867, "recall": 0.2689075630252101, "f1-score": 0.27839662307107993, "support": 1190}}
```

### Solution 1

Model: Llama3

Type: 8B-Original

Input: Multiple columns without column name

Output: Same number of column names (If the result number doesn't match, setting the result as mismatch)

Result: F1 score

```json
{"accuracy": 0.3878231859883236, "macro avg": {"precision": 0.17423837949782692, "recall": 0.13830692220391844, "f1-score": 0.14160185305424766, "support": 1199}, "weighted avg": {"precision": 0.464614318852247, "recall": 0.3878231859883236, "f1-score": 0.3816528355834458, "support": 1199}}
```

### Solution 2

Model: Llama3

Type: 70B-GPTQ

Input: Multiple columns without column name

Output: Same number of column names (If the result number doesn't match, setting the result as mismatch)

Result: F1 score

TODO

### Solution 3

Model: Llama3

Type: 70B-Fine-tunning

Input: Multiple columns without column name

Output: Same number of column names (If the result number doesn't match, setting the result as mismatch)

Result: F1 score

TODO

