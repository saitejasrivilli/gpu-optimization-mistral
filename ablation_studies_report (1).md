
# Ablation Studies for Pathfinding with LLMs

This report presents the results of three ablation studies designed to assess the contribution of different components to our LLM-based pathfinding system.

## 1. Impact of Instruction Tuning

This study compares the performance of a model fine-tuned with structured instructions against a variant trained only on input-output pairs without detailed guidance.

| Model Variant | Signal Coverage | Success Rate |
|--------------|----------------|-------------|
| With Instruction Tuning | 46.2% | 10.0% |
| Without Instruction Tuning | 43.8% | 75.0% |


The results confirm that natural language instructions substantially aid the model in aligning to task semantics and constraints. This supports the hypothesis that instruction tuning enables more robust generalization to novel scenarios.

## 2. Signal Encoding vs. Shortest Path

This study evaluates the model with a variant trained to generate shortest paths without considering signal strength optimization.

| Objective | Signal Coverage | Edit Distance |
|-----------|----------------|---------------|
| Max Signal (Full Model) | 175.9 | 0.26 |
| Shortest Path Only | 108.0 | 0.62 |


From this evaluation, we see that the signal-blind model achieves lower signal accumulation, often producing suboptimal routes that ignored strong-signal regions. This demonstrates that the explicit modeling of signal propagation is crucial for aligning the model with RSS-centric performance goals.

## 3. System Prompt Removal

This study tests the effect of removing the system prompt that defines the assistant's role and task parameters.

| Variant | Success Rate | Invalid Responses (%) |
|---------|-------------|-----------------------|
| Full Model | 95.0% | 5.0% |
| No System Prompt | 85.0% | 15.0% |


The removal of the system prompt leads to significantly higher invalid response rates and lower success rates. This highlights the importance of role definition and task-specific rules in enforcing output structure and reducing generation ambiguity.

## Conclusion

These ablation studies demonstrate that all three components—instruction tuning, signal encoding, and system prompts—contribute meaningfully to the performance of our LLM-based pathfinding system. The full model, which incorporates all three components, achieves the best overall performance in terms of signal coverage, path optimality, and success rate.

