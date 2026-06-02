
## 3. System Prompt Removal

We tested the effect of removing the system prompt, which contains the role definition and task-specific rules that guide the model's behavior. This ablation study directly measures the impact of clear instruction framing on model performance.

Without the system prompt, the model's outputs were more inconsistent across scenarios, with failure cases increasing significantly in environments with complex obstacle layouts or multiple transmitters. Success rates dropped from 95.0% to 85.0%, while invalid response rates increased from 5.0% to 15.0%.

This supports our hypothesis that the system prompt plays a critical role in enforcing task structure and reducing generation ambiguity. The clear definition of constraints (avoiding obstacles), movement rules (cardinal directions only), and optimization goals (maximizing signal) in the system prompt helps the model generate valid and effective solutions.

| Variant | Success Rate | Invalid Responses (%) |
|---------|-------------|----------------------|
| Full Model | 95.0% | 5.0% |
| No System Prompt | 85.0% | 15.0% |

The substantial decline in performance without the system prompt highlights the importance of providing models with appropriate context and constraints. This is particularly relevant for spatial reasoning tasks like pathfinding, where adherence to multiple constraints is critical for generating valid solutions.
