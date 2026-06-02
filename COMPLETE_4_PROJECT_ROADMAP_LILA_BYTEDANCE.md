# Complete 4-Project Roadmap: LILA + ByteDance

## Overview

**Your goal**: Get interviews at both LILA and ByteDance by building 4 complementary projects.

**LILA needs**: Causal reasoning + multimodal reasoning + scientific discovery capability  
**ByteDance needs**: RLHF understanding + model optimization + pre-training knowledge

**Timeline**: 12-14 weeks total (projects overlap)  
**GPU**: 3x A30 (efficient usage across all projects)

---

## Project Sequence & Timing

```
Week 1-4:   Project 1 (Causal Hypothesis Generator) - LILA
Week 3-6:   Project 2 (Multimodal Synthesis Reasoning) - LILA [overlap]
Week 5-9:   Project 3 (RLHF Fine-tuning) - ByteDance [parallel]
Week 9-13:  Project 4 (Mini Pre-training) - ByteDance [parallel]
Week 14:    Polish, GitHub, apply
```

Why this order:
- Projects 1&2 are more complex, start first
- Project 3 is easier (uses existing libraries), start middle
- Project 4 builds on 3, start last
- Weeks 5-9: All 4 in progress, heavy GPU usage
- Weeks 10-13: Projects 3&4 finish while 1&2 wind down

---

# PROJECT 1: Causal Hypothesis Generator for Chemistry

*Complete specification from previous document, summarized with key depth requirements*

## 1.1 Core Goal

Extract causal relationships from chemistry papers → build causal graph → generate novel hypotheses → validate against molecular data.

**What proves you can do this**:
- Extract causality from 500+ papers using fine-tuned LLM
- Build networkx DAG with 200+ nodes, 400+ edges
- Generate 20+ novel hypotheses using Pearl's do-calculus
- Achieve 65% validation rate (3.2× random baseline)

## 1.2 Data Collection

### Abstracts (500-1000)
- Source: ArXiv chemistry papers (last 2 years)
- Script: Use ArXiv API to fetch + filter by keyword (chemistry, materials, synthesis)
- Expected: 500 unique abstracts with author, title, link

### Ground Truth (10,000 molecules)
- Source: ChEMBL (free, public database)
- What to extract: SMILES + properties (melting point, solubility, yield, reaction rate)
- Processing: Use RDKit to compute additional descriptors

### Training Data (100-200 labeled examples)
- Manually label 100 abstracts for causal relations
- Format: `{abstract, [(cause1, effect1, confidence), ...]}`
- Tool: Spreadsheet or simple JSON annotation interface
- Time: 4-6 hours (can hire cheap annotators)

**Depth required for LILA**: You must understand how to:
- Parse scientific literature systematically
- Identify what's causal vs correlative
- Create rigorous training data for domain-specific tasks

## 1.3 Causal Extraction (LLM Fine-tuning)

### Fine-tune LLaMA 2 7B

**Setup**:
```
Base model: meta-llama/Llama-2-7b-hf
Method: LoRA (parameter-efficient)
Quantization: 4-bit (fit on A30)
Data: 100-200 labeled abstracts
Epochs: 3
Learning rate: 2e-4
```

**Key code patterns**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Quantize to fit on A30
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="models/causal_extractor",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,  # A30 supports bfloat16
    ),
    train_dataset=dataset,
    max_seq_length=512,
)

trainer.train()
```

**What this teaches you**:
- LoRA fine-tuning (parameter-efficient approach)
- How to handle 4-bit quantization
- Training loops and hyperparameter tuning
- Memory-constrained training on A30

**Depth for LILA**: They want to see you can:
- Fine-tune LLMs for specific domains (chemistry)
- Extract structured information from unstructured text
- Understand model capacity vs. data requirements

## 1.4 Causal Graph Construction

### Extract Relations from All 500 Papers

**Algorithm**:
```python
def extract_all_papers(model, tokenizer, paper_file, output_file):
    """
    For each paper:
    1. Prompt fine-tuned model with abstract
    2. Parse response into structured relations
    3. Deduplicate and aggregate across papers
    """
    
    papers = load_json(paper_file)
    results = []
    
    for paper in papers:
        prompt = f"""Extract causal relationships from this chemistry abstract:
{paper['abstract']}

Format: cause | effect | confidence (0-1)"""
        
        response = model.generate(
            tokenizer(prompt, return_tensors="pt"),
            max_new_tokens=256,
            temperature=0.7
        )
        
        # Parse response
        relations = parse_causal_output(response)
        results.append({
            'arxiv_id': paper['arxiv_id'],
            'relations': relations
        })
    
    return results
```

### Build Causal DAG

**Using networkx**:
```python
import networkx as nx

def build_causal_graph(extracted_relations):
    """
    Create directed acyclic graph from extracted relations
    
    Nodes: properties (e.g., "temperature", "yield")
    Edges: causal relations with confidence weights
    """
    
    G = nx.DiGraph()
    relation_counts = defaultdict(int)
    relation_confidence = defaultdict(list)
    
    # Aggregate relations across all papers
    for result in extracted_relations:
        for rel in result['relations']:
            cause = rel['cause']
            effect = rel['effect']
            conf = rel['confidence']
            
            key = (cause, effect)
            relation_counts[key] += 1
            relation_confidence[key].append(conf)
    
    # Add edges (only if observed 2+ times OR high confidence)
    for (cause, effect), count in relation_counts.items():
        avg_conf = sum(relation_confidence[(cause, effect)]) / len(relation_confidence[(cause, effect)])
        
        if count >= 2 or avg_conf >= 0.85:
            G.add_edge(cause, effect, weight=avg_conf, count=count)
    
    # Verify DAG property
    if not nx.is_directed_acyclic_graph(G):
        # Remove weakest edges in cycles
        for cycle in nx.simple_cycles(G):
            weakest = min(cycle, key=lambda e: G[e[0]][e[1]]['weight'])
            G.remove_edge(*weakest)
    
    return G
```

**Validation**:
```python
def validate_graph(G):
    """Check graph properties"""
    print(f"Nodes: {len(G.nodes())}")
    print(f"Edges: {len(G.edges())}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")
    
    # Find most influential causes
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    top_causes = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    top_effects = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"Top causes: {top_causes}")
    print(f"Top effects: {top_effects}")
```

**Metrics you need**:
- Graph size: 200+ nodes, 400+ edges
- Density: ~0.01-0.05 (sparse, as expected for chemistry)
- Longest path: 4-6 hops (reasonable causal chains)
- High-confidence edges: 150+ with confidence ≥0.9

**Depth for LILA**: This shows you can:
- Aggregate noisy information across many sources
- Build and validate graph structures
- Reason about domain semantics (what's a valid causal relation?)

## 1.5 Do-Calculus Reasoning (Pearl's Framework)

### Implement Causal Inference

**Key concept**: Pearl's do-calculus answers "what if I intervene?" questions.

```python
class CausalReasoner:
    def __init__(self, graph):
        self.graph = graph
    
    def find_causal_paths(self, source, target, max_depth=4):
        """Find all causal paths between two nodes"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_depth))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_confounders(self, source, target):
        """Find common causes (confounders) of source and target"""
        source_ancestors = set(nx.ancestors(self.graph, source))
        target_ancestors = set(nx.ancestors(self.graph, target))
        confounders = source_ancestors & target_ancestors
        return confounders
    
    def estimate_causal_effect(self, source, target, conditioning_set=None):
        """
        Estimate if causal effect from source to target is identifiable
        (i.e., can we estimate it without confounding?)
        """
        if conditioning_set is None:
            conditioning_set = set()
        
        # Find paths
        paths = self.find_causal_paths(source, target)
        
        # Find confounders
        confounders = self.get_confounders(source, target)
        confounders = confounders - conditioning_set
        
        # Effect is identified if all confounders are conditioned on
        identified = len(confounders) == 0
        
        # Calculate confidence (average confidence over paths)
        if paths:
            path_confidences = []
            for path in paths:
                path_conf = 1.0
                for i in range(len(path) - 1):
                    edge_conf = self.graph[path[i]][path[i+1]].get('weight', 0.8)
                    path_conf *= edge_conf
                path_confidences.append(path_conf)
            avg_confidence = sum(path_confidences) / len(path_confidences)
        else:
            avg_confidence = 0.0
        
        return {
            'source': source,
            'target': target,
            'paths': paths,
            'confounders': list(confounders),
            'identified': identified,
            'confidence': avg_confidence
        }
```

### Generate Hypotheses

**Algorithm**:
```python
def generate_hypotheses(reasoner, graph, top_n=20):
    """
    Generate novel hypotheses: pairs with indirect causal paths but no direct edge
    """
    nodes = list(graph.nodes())
    hypotheses = []
    
    for source, target in itertools.permutations(nodes, 2):
        # Skip if direct edge exists
        if graph.has_edge(source, target):
            continue
        
        # Check for indirect causal path
        result = reasoner.estimate_causal_effect(source, target)
        
        if result['paths']:
            hypothesis = {
                'source': source,
                'target': target,
                'hypothesis': f"Increasing {source} will increase {target}",
                'causal_paths': result['paths'],
                'num_paths': len(result['paths']),
                'path_length': min(len(p) for p in result['paths']),
                'confounders': result['confounders'],
                'confidence': result['confidence'],
                'identified': result['identified']
            }
            hypotheses.append(hypothesis)
    
    # Rank by: identified (prefer), then confidence, then path length
    hypotheses_ranked = sorted(
        hypotheses,
        key=lambda h: (h['identified'], h['confidence'], -h['path_length']),
        reverse=True
    )[:top_n]
    
    return hypotheses_ranked
```

**Proof of novelty**:
- 20+ hypotheses generated
- Each has causal path support (not just random)
- Most have confidence ≥0.7
- Can explain WHY each hypothesis is plausible

**Depth for LILA**: This demonstrates:
- Understanding of causal vs. correlational reasoning
- Ability to use formal frameworks (Pearl's do-calculus)
- Generation of novel scientific hypotheses
- **This is exactly what they need for scientific reasoning**

## 1.6 Validation Against Real Data

### Test Hypotheses on ChEMBL

```python
from scipy.stats import spearmanr

def validate_hypothesis(hypothesis, df_molecules, min_samples=10):
    """
    For each hypothesis, check if it matches real molecular data
    """
    
    source = hypothesis['source'].lower()
    target = hypothesis['target'].lower()
    
    # Find matching columns in molecular data
    source_col = find_column_match(df_molecules, source)
    target_col = find_column_match(df_molecules, target)
    
    if not source_col or not target_col:
        return {'validated': False, 'reason': 'No matching columns', 'score': 0}
    
    # Get data with both properties
    valid_data = df_molecules[[source_col, target_col]].dropna()
    
    if len(valid_data) < min_samples:
        return {'validated': False, 'reason': f'Insufficient data', 'score': 0}
    
    # Compute correlation
    corr, pvalue = spearmanr(valid_data[source_col], valid_data[target_col])
    
    # Hypothesis is correct if:
    # 1. Positive correlation
    # 2. Statistically significant (p < 0.05)
    is_correct = (corr > 0.1) and (pvalue < 0.05)
    
    # Score: weighted combination
    correlation_score = max(0, corr)
    significance_score = max(0, 1 - pvalue)
    causal_confidence = hypothesis.get('confidence', 0.5)
    
    validation_score = (
        0.4 * correlation_score +
        0.3 * significance_score +
        0.3 * causal_confidence
    )
    
    return {
        'validated': is_correct,
        'correlation': corr,
        'pvalue': pvalue,
        'n_samples': len(valid_data),
        'score': validation_score
    }
```

**Target metrics**:
- Validation rate: 65-70% (correct hypotheses)
- Baseline (random): ~30%
- Improvement: 2-3× over baseline

### Ablation Study

```python
def ablation_causal_vs_random(hypotheses, random_hypotheses, df_molecules):
    """
    Compare causal reasoning vs. random hypothesis generation
    """
    
    # Validate both sets
    causal_results = [validate_hypothesis(h, df_molecules) for h in hypotheses]
    random_results = [validate_hypothesis(h, df_molecules) for h in random_hypotheses]
    
    causal_rate = sum(1 for r in causal_results if r['validated']) / len(causal_results)
    random_rate = sum(1 for r in random_results if r['validated']) / len(random_results)
    
    improvement = (causal_rate - random_rate) / random_rate * 100
    
    print(f"Causal reasoning: {causal_rate:.1%} validation rate")
    print(f"Random baseline: {random_rate:.1%} validation rate")
    print(f"Improvement: {improvement:.1f}%")
    
    return {
        'causal_rate': causal_rate,
        'random_rate': random_rate,
        'improvement_percent': improvement
    }
```

**Depth for LILA**: Shows you can:
- Design rigorous evaluations
- Validate theoretical predictions against empirical data
- Prove your reasoning approach works better than baselines

## 1.7 Deliverables for Project 1

**Code**:
- `data/papers/` - 500+ abstracts
- `data/molecules/` - ChEMBL dataset
- `data/hypotheses/` - Generated + validated hypotheses
- `src/extract.py` - LLM-based extraction
- `src/graph.py` - Graph construction + validation
- `src/reasoning.py` - Do-calculus reasoning
- `src/validate.py` - Hypothesis validation
- `notebooks/` - Full pipeline walkthrough

**Results**:
```
Causal Extraction:
- 500 papers processed
- 1200 causal relations extracted
- 600 relations aggregated (multiple papers)

Causal Graph:
- 200+ nodes (chemical properties)
- 400+ edges (causal relations)
- Verified as DAG

Hypothesis Generation:
- 20 novel hypotheses generated
- Average confidence: 0.72
- All have causal path support

Validation:
- Validation rate: 68% (14/20 hypotheses match data)
- Random baseline: 30%
- Improvement: 2.3×
```

**Resume bullet**:
```
Implemented causal hypothesis generator for chemistry using fine-tuned LLaMA 2 
and Pearl's do-calculus; extracted 1200 causal relations from 500+ papers, 
constructed DAG with 200+ nodes, generated 20 novel hypotheses achieving 68% 
validation rate—2.3× improvement over random baseline.
```

**Timeline**: 3-4 weeks (Week 1-4)

---

# PROJECT 2: Multimodal Molecular Synthesis Reasoning System

*Builds on Project 1: uses same chemistry domain, adds multimodal + constraint-based reasoning*

## 2.1 Core Goal

Build system that predicts molecular synthesis pathways given:
- Molecule structure (image + SMILES)
- Constraints (temperature, safety, yield)

Output: Top-5 ranked pathways with reasoning traces.

**What proves you can do this**:
- Encode molecules via CLIP (multimodal vision+text)
- Fine-tune LLaMA 2 on 5K synthesis examples
- Validate on USPTO database: 68% top-5 accuracy
- 96% constraint satisfaction rate
- Interpretable reasoning traces explaining each pathway

## 2.2 Data Collection

### Synthesis Training Data (5000 reactions)

**Source**: USPTO Chemical Reactions (public, downloadable)

```python
import pandas as pd
from rdkit import Chem

def collect_uspto_data(output_file, num_reactions=5000):
    """
    Download from: https://github.com/rxn4chemistry/rxn-data
    Or Kaggle: Chemical Reactions from US Patents
    """
    
    df = pd.read_csv('uspto_reactions.csv')  # Download manually
    
    synthesis_data = []
    
    for idx, row in df.iterrows():
        reaction_smiles = row['reaction_smiles']
        
        try:
            # Parse reaction: "reactant1.reactant2>>product1.product2"
            parts = reaction_smiles.split('>>')
            if len(parts) != 2:
                continue
            
            reactants = parts[0].split('.')
            products = parts[1].split('.')
            
            synthesis_data.append({
                'target_molecule': products[0],  # Main product
                'other_products': products[1:],
                'reactants': reactants,
                'reaction_smiles': reaction_smiles,
                'patent_id': row.get('patent_id'),
                'year': row.get('year'),
            })
        
        except Exception:
            continue
        
        if len(synthesis_data) >= num_reactions:
            break
    
    with open(output_file, 'w') as f:
        json.dump(synthesis_data, f, indent=2)
    
    return synthesis_data
```

### Training Format for Fine-tuning

```json
{
  "instruction": "Analyze this chemical synthesis and generate reasoning about constraints",
  "input": "Target molecule (SMILES): CC(=O)Nc1ccc(O)cc1\nReactants: ...\nConstraints:\n- Max temperature: 300°C\n- Avoid toxic solvents\n- Min yield: 75%",
  "output": "Reasoning:\n1. Target is paracetamol with acetyl group\n2. Recommend mild conditions (< 300°C)\n3. Use non-toxic solvent (ethanol, water)\n4. Expected yield: 80-85%\n5. Constraints: ALL SATISFIED"
}
```

**Expected data**: 5000 synthesis examples

## 2.3 Molecule Encoding (Multimodal)

### CLIP-based Encoding

```python
from transformers import CLIPProcessor, CLIPModel
from rdkit import Chem
from rdkit.Chem import Draw
import torch

class MoleculeEncoder:
    """
    Encode molecules using CLIP: vision encoder sees molecule image,
    text encoder sees SMILES string
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def smiles_to_image(self, smiles, size=256):
        """Convert SMILES to image"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(size, size))
        return img
    
    def encode_molecule(self, smiles):
        """
        Encode molecule using both image and text
        Returns: combined embedding (image + text pooled)
        """
        
        # Generate image
        img = self.smiles_to_image(smiles)
        if img is None:
            return None
        
        # Process with CLIP
        inputs = self.processor(
            text=[smiles],
            images=[img],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pool image and text embeddings
        image_features = outputs.image_embeds  # [1, 512]
        text_features = outputs.text_embeds     # [1, 512]
        
        # Average (or concatenate for richer representation)
        combined = (image_features + text_features) / 2
        
        return combined.cpu().numpy()[0]  # [512]
    
    def encode_batch(self, smiles_list, batch_size=32):
        """Encode multiple molecules"""
        embeddings = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            
            for smiles in batch:
                emb = self.encode_molecule(smiles)
                if emb is not None:
                    embeddings.append(emb)
        
        return np.array(embeddings)
```

**Why CLIP for molecules?**
- Vision encodes structure (bonds, functional groups)
- Text (SMILES) encodes chemical identity
- Together: richer representation than either alone
- Directly addresses LILA/ByteDance requirement: "multimodal reasoning"

**Alternative: ChemBERTa**
- Faster (text-only)
- Chemistry-tuned
- But not multimodal

Use CLIP for the project (more impressive).

## 2.4 Fine-tune LLaMA 2 on Synthesis

Similar to Project 1, but with different data.

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

def finetune_synthesis_model():
    """Fine-tune LLaMA 2 on synthesis reasoning"""
    
    # Load base model (same as Project 1)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load training data
    dataset = load_dataset(
        'json',
        data_files='data/synthesis/synthesis_training.jsonl',
        split='train'
    )
    
    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir="models/synthesis_reasoner_lora",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            bf16=True,
            max_steps=1000,
        ),
        train_dataset=dataset,
        max_seq_length=1024,
    )
    
    trainer.train()
```

## 2.5 Constraint Parser

Parse natural language constraints into structured form.

```python
import re

class ConstraintParser:
    """
    Parse text like: "must synthesize at <300°C, avoid toxic solvents, yield >80%"
    Into: {max_temperature: 300, avoid_toxic_solvents: True, min_yield: 0.80}
    """
    
    def __init__(self):
        self.patterns = {
            'max_temperature': [
                r'(?:temp|temperature).*?<(\d+)',
                r'(?:below|under)\s*(\d+)\s*(?:°C|C)',
            ],
            'min_yield': [
                r'yield.*?(?:>|≥|at least)\s*(\d+)%',
            ],
            'avoid_toxic_solvents': [
                r'avoid\s+(?:toxic|dangerous)\s+solvents',
            ],
            'green_chemistry': [
                r'green\s+chemistry',
                r'environmentally.*?friendly',
            ],
            'aqueous_preferred': [
                r'aqueous.*?preferred',
                r'water.*?solvent',
            ],
        }
    
    def parse(self, constraint_text: str) -> dict:
        """Parse constraint text"""
        constraints = {}
        text_lower = constraint_text.lower()
        
        for constraint_name, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                
                if match:
                    if constraint_name in ['max_temperature', 'min_temperature']:
                        constraints[constraint_name] = int(match.group(1))
                    elif constraint_name in ['min_yield']:
                        constraints[constraint_name] = int(match.group(1)) / 100
                    else:
                        constraints[constraint_name] = True
                    break
        
        return constraints
    
    def validate(self, constraints: dict) -> bool:
        """Sanity check constraints"""
        if 'max_temperature' in constraints:
            if constraints['max_temperature'] <= 0:
                return False
        if 'min_yield' in constraints:
            if not 0 <= constraints['min_yield'] <= 1:
                return False
        return True
```

## 2.6 Synthesis Reasoner

```python
class SynthesisReasoner:
    """
    Main reasoning engine:
    1. Encode molecule
    2. Parse constraints
    3. Generate candidate pathways
    4. Rank by constraints + feasibility
    5. Generate reasoning traces
    """
    
    def __init__(self, model, tokenizer, encoder):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.constraint_parser = ConstraintParser()
    
    def generate_pathways(self, smiles, constraints_text, num_candidates=10):
        """
        Generate multiple synthesis pathway candidates
        """
        
        # Encode molecule (multimodal)
        mol_embedding = self.encoder.encode_molecule(smiles)
        
        # Parse constraints
        constraints = self.constraint_parser.parse(constraints_text)
        
        # Build prompt for LLM
        prompt = f"""Generate {num_candidates} different synthesis pathways.

Target molecule (SMILES): {smiles}

Constraints:
{self._format_constraints(constraints)}

For each pathway provide:
1. Step-by-step procedure
2. Temperature and time
3. Solvents and reagents
4. Expected yield
5. Constraint satisfaction check

Generate pathways:"""
        
        # Generate with sampling for diversity
        pathways = []
        
        for i in range(num_candidates):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7 + (i * 0.02),  # Increase for diversity
                    top_p=0.95,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            pathway = self._parse_pathway(response, i+1, constraints)
            
            if pathway:
                pathways.append(pathway)
        
        return pathways
    
    def rank_pathways(self, pathways, constraints):
        """
        Rank pathways by:
        1. Constraint satisfaction (most important)
        2. Estimated yield
        3. Simplicity
        4. Time efficiency
        """
        
        for pathway in pathways:
            score = 0.0
            
            # Constraint satisfaction (40%)
            constraint_score = 1.0 if pathway.get('constraints_satisfied') else 0.5
            score += 0.4 * constraint_score
            
            # Yield (30%)
            score += 0.3 * pathway.get('estimated_yield', 0.75)
            
            # Time (20%)
            time_hours = pathway.get('total_time_hours', 12)
            time_score = max(0, 1 - (time_hours / 24))
            score += 0.2 * time_score
            
            # Simplicity (10%)
            num_steps = len(pathway.get('steps', []))
            simplicity_score = max(0, 1 - (num_steps / 10))
            score += 0.1 * simplicity_score
            
            pathway['ranking_score'] = score
        
        return sorted(pathways, key=lambda x: x['ranking_score'], reverse=True)
```

## 2.7 Constraint Validation

```python
class ConstraintValidator:
    """
    Verify pathway satisfies chemical constraints
    """
    
    def __init__(self):
        self.toxic_solvents = [
            'benzene', 'carbon tetrachloride', 'toluene', 'xylene'
        ]
        self.green_solvents = [
            'water', 'ethanol', 'supercritical CO2', 'ionic liquids'
        ]
    
    def validate_temperature(self, max_temp_in_procedure, max_allowed):
        """Check temperature constraint"""
        if max_allowed is None:
            return True
        return max_temp_in_procedure <= max_allowed
    
    def validate_solvents(self, solvents, avoid_toxic=False):
        """Check solvent constraints"""
        if not avoid_toxic:
            return True
        for solvent in solvents:
            if solvent.lower() in self.toxic_solvents:
                return False
        return True
    
    def validate_yield(self, expected_yield, min_yield):
        """Check yield constraint"""
        if min_yield is None:
            return True
        return expected_yield >= min_yield
    
    def validate_pathway(self, pathway, constraints):
        """Comprehensive validation"""
        
        checks = {
            'temperature': self.validate_temperature(
                pathway.get('max_temp', 100),
                constraints.get('max_temperature')
            ),
            'solvents': self.validate_solvents(
                pathway.get('solvents', []),
                constraints.get('avoid_toxic_solvents', False)
            ),
            'yield': self.validate_yield(
                pathway.get('estimated_yield', 0.75),
                constraints.get('min_yield')
            ),
        }
        
        pathway['constraints_satisfied'] = all(checks.values())
        pathway['constraint_checks'] = checks
        
        return pathway
```

## 2.8 Reasoning Traces

Generate interpretable explanations for each pathway.

```python
def generate_reasoning_trace(pathway, constraints):
    """
    Create step-by-step reasoning explaining why this pathway works
    """
    
    trace = {
        'pathway_id': pathway.get('pathway_id'),
        'reasoning_steps': [],
        'constraint_checks': [],
        'confidence': 0.0
    }
    
    # Step 1: Analyze molecule
    trace['reasoning_steps'].append({
        'step': 1,
        'title': 'Analyze target structure',
        'content': f"Identify functional groups and synthetic accessibility",
    })
    
    # Step 2: Select starting materials
    trace['reasoning_steps'].append({
        'step': 2,
        'title': 'Choose starting materials',
        'content': f"Select readily available and cost-effective precursors",
    })
    
    # Step 3: Plan reaction sequence
    for i, step in enumerate(pathway.get('steps', []), start=3):
        trace['reasoning_steps'].append({
            'step': i,
            'title': f"Reaction {i-2}: {step.get('action', 'N/A')}",
            'content': f"{step.get('temperature', 'N/A')}°C, {step.get('time_hours', 1)}h",
        })
    
    # Constraint checks
    for constraint_name, passed in pathway.get('constraint_checks', {}).items():
        status = "✓ PASS" if passed else "✗ FAIL"
        trace['constraint_checks'].append({
            'constraint': constraint_name,
            'status': status,
        })
    
    # Confidence
    all_pass = all(c['status'] == "✓ PASS" for c in trace['constraint_checks'])
    trace['confidence'] = 0.9 if all_pass else 0.5
    
    return trace
```

## 2.9 Evaluation (USPTO)

```python
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

def evaluate_synthesis(predicted_pathway, ground_truth_reaction_smiles):
    """
    Compare predicted pathway against ground truth synthesis
    """
    
    predicted_product = predicted_pathway.get('target_molecule')
    
    # Parse ground truth
    parts = ground_truth_reaction_smiles.split('>>')
    if len(parts) < 2:
        return {'correct': False, 'similarity': 0.0}
    
    ground_truth_product = parts[1].split('.')[0]
    
    # Molecular similarity
    try:
        mol1 = Chem.MolFromSmiles(predicted_product)
        mol2 = Chem.MolFromSmiles(ground_truth_product)
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        similarity = TanimotoSimilarity(fp1, fp2)
    except:
        similarity = 0.0
    
    return {
        'correct': similarity >= 0.9,
        'similarity': similarity,
    }

def evaluate_top_k(ranked_pathways, ground_truth, k=5):
    """
    Check if correct pathway is in top-k
    """
    for i, pathway in enumerate(ranked_pathways[:k]):
        result = evaluate_synthesis(pathway, ground_truth)
        if result['correct']:
            return {'found': True, 'rank': i+1}
    
    return {'found': False, 'rank': None}
```

**Target metrics**:
- Top-1 accuracy: 45-55%
- Top-5 accuracy: 68-75%
- Constraint satisfaction: 96%+
- Reasoning trace quality: 8/10 from expert review

### Ablation: Constraints Impact

```python
def ablation_constraints():
    """
    Compare accuracy with vs. without constraint checking
    """
    
    # With constraints
    ranked_with = reasoner.rank_pathways(pathways, constraints)
    accuracy_with = evaluate_top_k(ranked_with, ground_truth, k=5)
    
    # Without constraints (rank by yield only)
    ranked_without = sorted(
        pathways,
        key=lambda x: x.get('estimated_yield', 0.5),
        reverse=True
    )
    accuracy_without = evaluate_top_k(ranked_without, ground_truth, k=5)
    
    print(f"With constraints: {accuracy_with['found']} (rank {accuracy_with['rank']})")
    print(f"Without constraints: {accuracy_without['found']} (rank {accuracy_without['rank']})")
    # Expected: constraints improve accuracy 10-15%
```

## 2.10 Deliverables for Project 2

**Code**:
- `src/encoder.py` - CLIP-based molecule encoding
- `src/constraint_parser.py` - Parse constraints
- `src/reasoner.py` - Synthesis reasoning engine
- `src/validator.py` - Constraint validation
- `src/reasoning_traces.py` - Generate interpretable traces
- `src/evaluate.py` - USPTO evaluation

**Results**:
```
Encoding:
- 5000 molecules encoded via CLIP
- Embedding dimension: 512

Fine-tuning:
- 5000 synthesis examples
- 3 epochs training
- Model: Llama 2 7B + LoRA

Inference:
- Top-1 accuracy: 52%
- Top-5 accuracy: 72%
- Constraint satisfaction: 96%
- Avg reasoning trace quality: 8.2/10

Ablation:
- With constraints: 72% top-5
- Without constraints: 58% top-5
- Improvement: 14%
```

**Resume bullet**:
```
Developed multimodal molecular synthesis reasoner integrating CLIP-based molecule 
encoding + fine-tuned LLaMA 2 + constraint satisfaction; achieved 72% top-5 accuracy 
on USPTO with 96% constraint adherence; generated interpretable reasoning traces; 
constraint-aware ranking improved accuracy 14% over unconstrained baseline.
```

**Timeline**: 3-4 weeks (Week 3-6, overlapping with Project 1)

---

# PROJECT 3: RLHF Fine-tuning with Preference Alignment

*For ByteDance: shows you understand preference alignment, not just supervised fine-tuning*

## 3.1 Core Goal

Train model using full RLHF pipeline:
1. Fine-tune base model on reasoning task (SFT)
2. Train reward model to predict preferences
3. Use PPO to align model with reward function
4. Show quantified improvement

**What proves you can do this**:
- SFT baseline: 62% accuracy
- After RLHF: 78% accuracy
- Preference score (human evaluation): 6.2/10 → 8.1/10
- Understand entire RLHF pipeline

## 3.2 Data Collection

### Reasoning Tasks (1000-2000 examples)

Pick domain with clear correctness: math, code logic, physics reasoning.

```python
def create_reasoning_dataset():
    """
    Collect reasoning problems with step-by-step solutions
    Sources:
    - MATH dataset (MIT-KBQA)
    - CodeForces problems
    - Science QA
    """
    
    data = {
        'math': load_math_dataset(),  # 1000 problems
        'code': load_codewars_problems(),  # 500 problems
    }
    
    dataset = []
    for problem_type, problems in data.items():
        for problem in problems:
            dataset.append({
                'problem': problem['question'],
                'solution': problem['solution'],
                'answer': problem['answer'],
                'difficulty': problem.get('difficulty', 'medium'),
                'type': problem_type
            })
    
    return dataset  # ~1500 examples
```

### Preference Pairs (500-1000)

Create human preferences: which response is better?

```
Problem: "What is 7 + 5?"

Response A: "7 + 5 = 12"
Response B: "7 + 5 equals 12. This is simple addition of two positive integers."

Preference: B (more detailed, clearer reasoning)
```

You can generate synthetic preferences via:
- Length (longer = more explanation)
- Correctness + explanation
- Step-by-step reasoning vs. direct answer

```python
def create_preference_pairs(sft_model, dataset, num_pairs=500):
    """
    Generate model responses, then create preference pairs
    """
    
    preference_pairs = []
    
    for example in dataset[:num_pairs]:
        # Generate 2 responses with different temperatures
        response_a = generate(sft_model, example['problem'], temperature=0.5)
        response_b = generate(sft_model, example['problem'], temperature=0.9)
        
        # Create preference (deterministic or heuristic-based)
        # Heuristic: longer + contains step-by-step = preferred
        pref_score_a = score_response(response_a, example['answer'])
        pref_score_b = score_response(response_b, example['answer'])
        
        preferred = 'a' if pref_score_a >= pref_score_b else 'b'
        
        preference_pairs.append({
            'problem': example['problem'],
            'response_a': response_a,
            'response_b': response_b,
            'preferred': preferred
        })
    
    return preference_pairs
```

## 3.3 SFT Baseline

Standard supervised fine-tuning on reasoning tasks.

```python
def finetune_sft(dataset, output_dir):
    """
    Stage 1: Supervised fine-tuning
    """
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    # Train
    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            bf16=True,
            learning_rate=2e-4,
            max_steps=500,
        ),
        train_dataset=dataset,
    )
    
    trainer.train()
    model.save_pretrained(f"{output_dir}/sft_model")
    
    return model
```

**Evaluate SFT**:
```python
def evaluate_sft(model, test_set):
    """
    Compute accuracy on reasoning tasks
    """
    correct = 0
    for example in test_set:
        response = generate(model, example['problem'])
        if check_correctness(response, example['answer']):
            correct += 1
    
    accuracy = correct / len(test_set)
    return accuracy

sft_accuracy = evaluate_sft(sft_model, test_set)
print(f"SFT Accuracy: {sft_accuracy:.1%}")  # Expected: 60-65%
```

## 3.4 Reward Model Training

Train model to predict which response is better.

```python
import torch.nn as nn

class RewardModel(nn.Module):
    """
    Predicts preference score for a response
    Input: response text
    Output: scalar reward (0-1)
    """
    
    def __init__(self, base_model_name="meta-llama/Llama-2-7b-hf"):
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            output_hidden_states=True,
        )
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add reward head
        hidden_size = self.model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        
        # Use last hidden state
        last_hidden = outputs.hidden_states[-1]
        
        # Pool (mean of all tokens)
        pooled = last_hidden.mean(dim=1)
        
        # Reward score
        reward = self.reward_head(pooled)
        
        return reward

def train_reward_model(preference_pairs, num_epochs=3):
    """
    Train reward model on preference pairs
    """
    
    reward_model = RewardModel()
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for pair in preference_pairs:
            # Encode both responses
            response_a_ids = tokenizer(pair['response_a'], return_tensors='pt')
            response_b_ids = tokenizer(pair['response_b'], return_tensors='pt')
            
            # Get rewards
            reward_a = reward_model(**response_a_ids)
            reward_b = reward_model(**response_b_ids)
            
            # Loss: reward for preferred should be > non-preferred
            if pair['preferred'] == 'a':
                loss = torch.nn.functional.relu(reward_b - reward_a + 1.0)
            else:
                loss = torch.nn.functional.relu(reward_a - reward_b + 1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return reward_model
```

**Evaluate reward model**:
```python
def evaluate_reward_model(reward_model, preference_pairs):
    """
    Check if reward model correctly ranks preferences
    """
    
    correct = 0
    for pair in preference_pairs:
        reward_a = reward_model(pair['response_a'])
        reward_b = reward_model(pair['response_b'])
        
        predicted_preferred = 'a' if reward_a > reward_b else 'b'
        
        if predicted_preferred == pair['preferred']:
            correct += 1
    
    accuracy = correct / len(preference_pairs)
    return accuracy

reward_accuracy = evaluate_reward_model(reward_model, preference_pairs)
print(f"Reward Model Accuracy: {reward_accuracy:.1%}")  # Expected: 70-80%
```

## 3.5 PPO Training

Use reward model to fine-tune SFT model via reinforcement learning.

```python
from trl import PPOTrainer, PPOConfig

def train_ppo(sft_model, reward_model, dataset, num_steps=1000):
    """
    PPO: Proximal Policy Optimization
    
    Objective: Maximize E[reward] while staying close to SFT model (KL penalty)
    """
    
    ppo_config = PPOConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        learning_rate=1e-4,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
        ppo_epochs=4,
        target_kl=0.1,  # KL penalty (stay close to SFT)
        seed=42,
        output_dir="models/ppo_output",
    )
    
    trainer = PPOTrainer(
        config=ppo_config,
        model=sft_model,
        ref_model=sft_model,  # Reference model (SFT)
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=torch.optim.AdamW(sft_model.parameters(), lr=ppo_config.learning_rate),
    )
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 100,
    }
    
    for step in range(num_steps):
        # Generate responses
        query_tensors = dataset['input_ids']
        response_tensors = trainer.generate(
            query_tensors,
            **generation_kwargs,
        )
        
        # Compute rewards
        responses = tokenizer.batch_decode(response_tensors)
        rewards = [reward_model(r) for r in responses]
        
        # PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)
        
        if step % 100 == 0:
            print(f"Step {step}: Reward mean = {stats['reward_mean']:.3f}, KL = {stats['kl']:.3f}")
    
    return trainer.model
```

## 3.6 Evaluation

```python
def evaluate_ppo_model(ppo_model, test_set):
    """
    Evaluate final model on reasoning tasks
    """
    
    correct = 0
    for example in test_set:
        response = generate(ppo_model, example['problem'])
        if check_correctness(response, example['answer']):
            correct += 1
    
    accuracy = correct / len(test_set)
    return accuracy

# Human preference scoring (manual or heuristic)
def evaluate_preference(model, test_set):
    """
    Score responses on helpfulness + clarity
    """
    
    scores = []
    for example in test_set:
        response = generate(model, example['problem'])
        
        # Heuristic: has steps + correct answer + clear = high score
        has_steps = "step" in response.lower() or "first" in response.lower()
        is_correct = check_correctness(response, example['answer'])
        is_clear = len(response) > 200  # Rough proxy
        
        score = int(has_steps) * 3 + int(is_correct) * 4 + int(is_clear) * 2
        score = min(10, score)  # Cap at 10
        
        scores.append(score)
    
    return sum(scores) / len(scores)

# Compare
sft_accuracy = evaluate_sft(sft_model, test_set)
ppo_accuracy = evaluate_ppo_model(ppo_model, test_set)
sft_preference = evaluate_preference(sft_model, test_set)
ppo_preference = evaluate_preference(ppo_model, test_set)

print(f"SFT: {sft_accuracy:.1%} accuracy, {sft_preference:.1f}/10 preference")
print(f"PPO: {ppo_accuracy:.1%} accuracy, {ppo_preference:.1f}/10 preference")
# Expected: +15-20% improvement
```

## 3.7 Deliverables

**Results**:
```
SFT Baseline:
- Accuracy: 62%
- Preference score: 6.2/10

RLHF (PPO) Final:
- Accuracy: 78%
- Preference score: 8.1/10
- Improvement: +16% accuracy, +1.9/10 preference

Training metrics:
- PPO steps: 1000
- Reward model accuracy: 75%
- KL divergence: 0.08 (stayed close to SFT)
```

**Resume bullet**:
```
Implemented RLHF pipeline: fine-tuned LLaMA 2 on reasoning tasks (62% SFT baseline), 
trained reward model on preference pairs (75% accuracy), applied PPO with KL penalty 
achieving 78% accuracy (+16% improvement); model preference score: 8.1/10 vs 6.2/10 SFT.
```

**Timeline**: 2-3 weeks (Week 5-7)

---

# PROJECT 4: Mini Pre-training (1B Parameter Model)

*Final project for ByteDance: shows you understand pre-training from scratch*

## 4.1 Core Goal

Train a 1B parameter model from scratch on public data.

**Why 1B?**
- Trainable on 3x A30 in 3 weeks
- Large enough to show you understand training dynamics
- Small enough to be feasible
- Shows you can scale (can think about larger training)

**What proves you can do this**:
- Train model from random initialization
- Understand data pipeline (tokenization, batching)
- Handle training stability (loss curves, convergence)
- Distributed training across multiple GPUs
- Compare against open 1B models
- Reach loss parity with existing models

## 4.2 Model Architecture

Simple transformer (or use existing small architecture).

```python
from transformers import GPT2Config, GPT2LMHeadModel

def create_1b_model():
    """
    1B parameter model: ~48 layers, 768 hidden dim
    """
    
    config = GPT2Config(
        vocab_size=50257,
        n_positions=2048,  # Context length
        n_embd=768,        # Hidden dim
        n_layer=48,        # 48 layers
        n_head=12,         # 12 attention heads
        activation_function="gelu",
        dropout=0.1,
    )
    
    model = GPT2LMHeadModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params / 1e9:.2f}B parameters")
    
    return model
```

## 4.3 Data Pipeline

Prepare 10B tokens of training data.

```python
def create_pretraining_dataset(data_sources, output_dir):
    """
    Collect and tokenize pretraining data
    
    Sources:
    - C4 (160GB, English web text)
    - WikiText-103 (raw Wikipedia)
    - GitHub (code)
    - Books (project Gutenberg)
    
    For quick start: use HF datasets
    """
    
    from datasets import load_dataset
    
    # Load datasets
    datasets = [
        load_dataset("wikitext", "wikitext-103-v1", split="train"),
        load_dataset("openwebtext", split="train"),  # Subset
    ]
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
    
    # Process
    tokenized_datasets = [
        ds.map(
            tokenize_function,
            batched=True,
            remove_columns=ds.column_names
        )
        for ds in datasets
    ]
    
    # Concatenate
    train_dataset = concatenate_datasets(tokenized_datasets)
    
    # Save
    train_dataset.save_to_disk(output_dir)
    
    return train_dataset
```

**Data requirements**:
- 10B tokens (target)
- Vocabulary: 50K tokens
- Batch size: 128 (distributed)
- Sequence length: 2048

## 4.4 Distributed Training Setup

Train across 3x A30 GPUs.

```python
from torch.distributed import launch
from transformers import Trainer, TrainingArguments

def train_pretrain(model, dataset):
    """
    Distributed training using Hugging Face Trainer
    """
    
    training_args = TrainingArguments(
        output_dir="models/pretrain_1b",
        
        # Training
        num_train_epochs=1,  # One epoch over dataset
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,  # Effective batch: 32 * 4 * 3 = 384
        
        # Learning
        learning_rate=1e-4,
        warmup_steps=2000,
        weight_decay=0.01,
        
        # Optimization
        optim="adamw_torch",
        fp16=True,
        max_grad_norm=1.0,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        
        # Distributed
        local_rank=0,  # Set by launcher
        ddp_find_unused_parameters=False,
        
        # Logging
        logging_steps=100,
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not MLM
        ),
    )
    
    trainer.train()
    model.save_pretrained("models/pretrain_1b_final")
```

**Training on 3x A30**:
```bash
# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    train_script.py
```

## 4.5 Training Dynamics Monitoring

Track key metrics during training.

```python
def monitor_training(trainer):
    """
    Watch training curves for stability indicators
    """
    
    metrics_to_track = {
        'loss': [],
        'learning_rate': [],
        'grad_norm': [],
        'batch_size': [],
    }
    
    for step in range(num_training_steps):
        # Log metrics
        metrics_to_track['loss'].append(trainer.state.log_history[-1]['loss'])
        
        # Check for divergence
        if len(metrics_to_track['loss']) > 100:
            recent_loss = metrics_to_track['loss'][-100:]
            if max(recent_loss) / min(recent_loss) > 2.0:
                print("WARNING: Large loss spike detected")
        
        # Log training curve
        if step % 100 == 0:
            print(f"Step {step}: Loss = {metrics_to_track['loss'][-1]:.3f}")
```

**Expected loss curve**:
```
Step 0:      Loss = 10.82 (log(50257) random baseline)
Step 100:    Loss = 8.45
Step 500:    Loss = 6.20
Step 1000:   Loss = 5.10
Step 5000:   Loss = 4.12
...final:    Loss = 3.45-3.80 (parity with GPT2-small baseline)
```

## 4.6 Evaluation & Benchmarking

Compare against existing 1B models.

```python
def evaluate_pretrained_model(model, test_dataset):
    """
    Compute perplexity on held-out test set
    """
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_dataset:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, perplexity

# Compare models
def benchmark_1b_models():
    """
    Compare your pretrained 1B model against:
    - GPT2-small (124M, smaller but published baseline)
    - DistilGPT2
    - Custom 1B model
    """
    
    models = {
        'your_1b_pretrained': your_model,
        'gpt2': load_model('gpt2'),
        'distilgpt2': load_model('distilgpt2'),
    }
    
    results = {}
    for name, model in models.items():
        loss, ppl = evaluate_pretrained_model(model, test_set)
        results[name] = {'loss': loss, 'perplexity': ppl}
    
    print("Perplexity Comparison:")
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['perplexity']):
        print(f"  {name}: {metrics['perplexity']:.2f}")
    
    return results
```

**Target benchmark**:
```
Your 1B model:        Perplexity = 22.5
GPT2-small (124M):    Perplexity = 24.3
DistilGPT2 (82M):     Perplexity = 28.1

Interpretation: Your 1B model is better than smaller models
(proves training helped, not just luck)
```

## 4.7 Ablation: Training Impact

```python
def ablation_training_data():
    """
    Show impact of different amounts of training data
    """
    
    data_sizes = [1, 2, 5, 10]  # Billion tokens
    
    for size in data_sizes:
        # Train model on 'size' B tokens
        model = train_on_data(size)
        loss, ppl = evaluate(model, test_set)
        
        print(f"Training tokens: {size}B -> Perplexity: {ppl:.2f}")
    
    # Expected: diminishing returns as data increases
    # 1B tokens → ppl ~30
    # 5B tokens → ppl ~25
    # 10B tokens → ppl ~22.5
```

## 4.8 Scaling Analysis

Show you understand pre-training at scale.

```python
def compute_compute_budget():
    """
    Analyze FLOPs and training time for your pretrain
    """
    
    num_params = 1e9  # 1B
    num_tokens = 10e9  # 10B
    
    # Rough FLOPs: 6 * num_params * num_tokens
    # (forward + backward pass)
    flops = 6 * num_params * num_tokens
    
    # Training time on 3x A30 (teraflops per GPU)
    a30_tflops = 326  # Peak FP32
    gpu_count = 3
    
    training_hours = flops / (gpu_count * a30_tflops * 1e12) * 1.5  # 1.5x for overhead
    training_days = training_hours / 24
    
    print(f"Compute budget:")
    print(f"  FLOPs: {flops / 1e18:.2f} ExaFLOPs")
    print(f"  Training time (3x A30): {training_days:.1f} days ({training_hours:.0f} hours)")
    print(f"  GPU-hours: {gpu_count * training_hours:.0f}")
    
    # Projection to larger model
    params_7b = 7e9
    tokens_1t = 1e12
    flops_7b = 6 * params_7b * tokens_1t
    training_days_7b = (flops_7b / (gpu_count * a30_tflops * 1e12)) * 1.5 / 24
    
    print(f"\nExtrapolation to 7B parameters, 1T tokens:")
    print(f"  Estimated training time (3x A30): {training_days_7b:.0f} days")
    print(f"  Scaling factor vs 1B: {training_days_7b / training_days:.0f}×")
```

## 4.9 Deliverables

**Results**:
```
Pre-training Summary:
- Model size: 1.04B parameters
- Training data: 10B tokens (C4 + WikiText + OpenWebText)
- Training time: 15 days on 3x A30
- GPU-hours: 1080 GPU-hours

Loss curves:
- Initial: 10.82 (random baseline)
- After 1B tokens: 5.50
- After 5B tokens: 4.20
- Final (10B tokens): 3.62

Benchmarks:
- Your model perplexity: 22.5
- GPT2-small (124M): 24.3
- Improvement: 7.4% better than smaller baseline

Training stability:
- No divergence observed
- Gradient norms stable
- Learning rate schedule: warmup → constant → decay
- KL divergence maintained at 0.05

Scaling analysis:
- 1B model on 10B tokens: 15 days
- Projected 7B model on 1T tokens: 450 days (30 months)
- (Aligns with industry training times)
```

**Resume bullet**:
```
Trained 1B parameter language model from scratch on 10B tokens using distributed 
training across 3x A30 GPUs; achieved 22.5 perplexity (7.4% better than GPT2-small 
baseline); monitored training dynamics, loss curves, gradient stability; computed 
scaling laws showing 7B model on 1T tokens would require ~30 months on same hardware.
```

**Timeline**: 3-4 weeks (Week 9-13)

---

# COMPLETE TIMELINE & GPU ALLOCATION

## Week-by-Week Plan

```
Week 1:
  - P1: Data collection (arxiv papers, ChEMBL molecules)
  - P1: Manual labeling (causal relations)
  
Week 2:
  - P1: LLM fine-tuning (extract causal relations)
  - P1: Extract relations from 500 papers
  - GPU: 1x A30 for LLM training
  
Week 3:
  - P1: Build causal graph, generate hypotheses
  - P2: Data collection (USPTO reactions, SMILES)
  - GPU: 1x A30 for P1 analysis
  
Week 4:
  - P1: Hypothesis validation on ChEMBL
  - P1: Ablation study (with/without causality)
  - P2: Prepare synthesis fine-tuning data
  - GPU: 1x A30 for P1, P2 prep
  
Week 5:
  - P1: Polish, documentation, GitHub
  - P2: Fine-tune LLaMA 2 on synthesis (start)
  - P3: Create reasoning dataset, preference pairs
  - GPU: 1x A30 for P2, P3 prep
  
Week 6:
  - P2: Synthesis reasoning pipeline (inference)
  - P2: Constraint validation + ranking
  - P3: Train SFT baseline on reasoning
  - GPU: 1x A30 for P2, 1x A30 for P3
  
Week 7:
  - P2: Evaluation on USPTO, ablation study
  - P3: Train reward model on preferences
  - GPU: 1x A30 for P2, 1x A30 for P3
  
Week 8:
  - P2: Polish, GitHub
  - P3: PPO training (expensive, needs 2-3 GPUs)
  - GPU: 2-3x A30 for PPO
  
Week 9:
  - P3: PPO training continues
  - P3: Evaluation + comparison (SFT vs PPO)
  - P4: Prepare pretraining data (tokenization)
  - GPU: 2x A30 for P3, 1x A30 for P4 prep
  
Week 10:
  - P3: Polish, GitHub
  - P4: Distributed training launch (1B model)
  - GPU: 3x A30 for P4 (all in use)
  
Week 11-12:
  - P4: Pre-training loop
  - Monitor loss curves, training stability
  - GPU: 3x A30 for P4 (continuous)
  
Week 13:
  - P4: Final evaluation + benchmarking
  - P4: Compute scaling analysis
  - GPU: 1x A30 for P4 eval
  
Week 14:
  - Polish all 4 projects
  - Create GitHub repos with READMEs
  - Update resume with all 4 projects
  - Prepare applications
```

## GPU Utilization Summary

```
Week 1-4:   P1 (1x A30 avg)
Week 5-6:   P1 finish + P2 start (1x A30 avg)
Week 7-8:   P2 finish + P3 start (1.5x A30 avg)
Week 9:     P3 + P4 prep (2x A30 avg)
Week 10-12: P4 (3x A30 continuous)
Week 13-14: Finishing touches (0.5x A30 avg)

Peak: Week 10-12 (all 3x A30 on P4)
Valleys: Week 1, 13-14 (can use for other tasks)
```

---

# INTERVIEW NARRATIVES

## For LILA

**Your story**:

"I built two projects demonstrating scientific reasoning capability. 

**Project 1** shows I can extract causal relationships from scientific literature using domain-specific LLM fine-tuning, construct formal causal graphs, and use Pearl's do-calculus to generate novel hypotheses. Crucially, I validated these hypotheses against real molecular data—achieving 68% validation rate, 2.3× better than random baseline.

**Project 2** extends this to multimodal reasoning: encoding molecules both as images (via CLIP) and SMILES text, fine-tuning language models for scientific domain tasks, and importantly, reasoning under constraints. This shows I understand that scientific discovery isn't just prediction—it's satisfying real-world requirements (temperature limits, safety, yield).

Together, these projects prove I can:
- Read and extract structure from scientific literature
- Implement formal reasoning frameworks (do-calculus)
- Think about multiple modalities (vision + text)
- Validate theoretical predictions against empirical data
- Reason under domain-specific constraints

This is exactly what scientific reasoning requires."

## For ByteDance

**Your story**:

"I built two complementary projects showing understanding of the LLM development pipeline.

**Project 3** demonstrates preference alignment: I fine-tuned a model with standard SFT (62% accuracy), then trained a reward model to predict human preferences, and used PPO to align the model with that reward signal. Result: 78% accuracy (+16%), preference score from 6.2/10 to 8.1/10. This shows I understand the full RLHF pipeline—not just supervised learning.

**Project 4** is the foundational piece: I trained a 1B parameter language model from scratch on 10B tokens using distributed training across multiple GPUs. I monitored training dynamics, loss curves, gradient stability. I benchmarked against existing models and achieved parity. I computed scaling laws.

Combined, I can:
- Understand model training from initialization to convergence
- Apply preference alignment techniques for capability improvement
- Work with distributed training infrastructure
- Monitor and debug training stability
- Make scale-up decisions based on compute budgets

This shows I'm not just an inference engineer—I understand how Doubao was built."

---

# FINAL CHECKLIST

Before applying:

**Project 1 (Causal)**:
- [ ] 500+ papers collected
- [ ] 100+ abstracts manually labeled
- [ ] LLM fine-tuned on causal extraction
- [ ] Relations extracted from all papers
- [ ] Causal graph built (200+ nodes)
- [ ] 20+ hypotheses generated
- [ ] Validation on ChEMBL (65%+ rate)
- [ ] Ablation study complete
- [ ] Code on GitHub with README
- [ ] Results documented

**Project 2 (Synthesis)**:
- [ ] 5000 synthesis examples collected
- [ ] CLIP molecule encoding working
- [ ] LLaMA 2 fine-tuned on synthesis
- [ ] Constraint parser implemented
- [ ] Pathway ranking system working
- [ ] Reasoning traces generated
- [ ] Evaluated on USPTO (68% top-5)
- [ ] Constraint satisfaction 96%+
- [ ] Ablation study (with/without constraints)
- [ ] Code on GitHub with README
- [ ] Results documented

**Project 3 (RLHF)**:
- [ ] Reasoning dataset collected (1000+ examples)
- [ ] Preference pairs created (500+ pairs)
- [ ] SFT baseline trained (62% accuracy)
- [ ] Reward model trained (75%+ accuracy)
- [ ] PPO training completed (1000+ steps)
- [ ] Final model evaluated (78% accuracy)
- [ ] Before/after comparison documented
- [ ] Code on GitHub with README
- [ ] Results documented

**Project 4 (Pre-training)**:
- [ ] Data prepared (10B tokens)
- [ ] 1B model created
- [ ] Distributed training working on 3x A30
- [ ] Training completed (10-15 days)
- [ ] Loss curves look good
- [ ] Benchmarked against baselines
- [ ] Scaling analysis computed
- [ ] Code on GitHub with README
- [ ] Results documented

**Resume**:
- [ ] All 4 projects listed with bullet points
- [ ] Quantified results for each
- [ ] GitHub links included

**GitHub**:
- [ ] 4 clean repositories (one per project)
- [ ] Each has detailed README
- [ ] Code is well-organized and commented
- [ ] Results reproduced in notebooks
- [ ] No placeholder TODOs

**Applications**:
- [ ] LILA: Tailored resume (emphasize Projects 1&2)
- [ ] ByteDance: Tailored resume (emphasize Projects 3&4)
- [ ] Cover letter/statement for each role
- [ ] GitHub links in application

---

# FAQ & TROUBLESHOOTING

## What if pre-training diverges (loss goes to NaN)?

**Signs**: Loss suddenly becomes NaN or Inf after stable training

**Fixes** (in order):
1. Lower learning rate (divide by 2-10)
2. Reduce batch size (memory issues)
3. Gradient clipping (already in code: max_grad_norm=1.0)
4. Check data pipeline (NaN in input data?)
5. Warmup longer (increase warmup_steps)

**Prevention**: Monitor loss curve every 100 steps. If loss starts rising after initial decrease, act fast.

## What if reward model doesn't learn?

**Sign**: Reward model accuracy stuck at 50% (random chance)

**Fixes**:
1. Check preference pairs are clearly different (longer/shorter, correct/wrong)
2. Try different architecture (simpler reward head)
3. Higher learning rate for reward model (1e-3 instead of 1e-4)
4. More training data (create 1000+ pairs)

## What if projects take longer than expected?

**Compress by**:
1. Project 1: Use fewer papers (300 instead of 500)
2. Project 2: Use fewer synthesis examples (2000 instead of 5000)
3. Project 3: Skip ablation, just show final RLHF result
4. Project 4: Train on 5B tokens instead of 10B (faster, still shows understanding)

**Reality**: Week timelines are optimistic. Add 20-30% buffer.

## Should I apply before all 4 are done?

**No.** Wait until:
- At least Projects 1&2 are done (apply to LILA)
- All 4 are done (apply to ByteDance)

Early applications waste your one shot. Timing > speed.

---

# SUCCESS METRICS

**You'll know you're ready when**:

✓ LILA projects have:
- Quantified before/after metrics
- Real data validation
- Ablation studies showing your approach is better

✓ ByteDance projects have:
- Working training loops (no errors for hours)
- Clear loss curves showing convergence
- Benchmark comparisons against baselines

✓ All projects have:
- Clean GitHub repos with READMEs
- Reproducible results in notebooks
- Clear, honest documentation of what worked and what didn't

✓ You can explain:
- Why you built each project (what gap does it close?)
- How it differs from existing work
- What you learned in the process

**If any of these are missing, you're not ready yet.**

---

# GOOD LUCK

You're about to spend 12-14 weeks building four ambitious projects. This is hard. You'll hit bugs, divergences, dead ends.

But here's the thing: **if you get through this roadmap, you're not a "candidate who's been doing ML stuff."** You're someone who:
- Can read research papers and extract knowledge
- Can build reasoning systems from first principles
- Can fine-tune models at scale
- Can train models from scratch
- Can evaluate rigorously
- Can ship working systems

That's the difference between "interesting resume" and "we need to talk to this person."

Go build.
