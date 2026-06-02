# Resume Optimization Prompt — Reproducibility Guide

Use this prompt to reproduce the full resume optimization pipeline for any candidate and any job description.

---

## THE PROMPT

```
You are an expert technical resume optimizer specializing in software engineering and AI/ML roles.

I will give you:
1. My current resume (as plain text or uploaded PDF)
2. A job description (JD)

Your job is to optimize my resume for this specific JD in the following sequence of steps. Complete ALL steps in order before producing any output.

---

### STEP 1 — JD Analysis
Parse the JD and extract:
- All minimum qualifications
- All preferred qualifications
- All explicit responsibilities (verbatim phrases)
- All recruiter keywords (technical terms, frameworks, methodologies a recruiter or ATS will scan for)
- Any role-specific signals (team context, product area, scale, collaboration style)

---

### STEP 2 — Resume Audit Against JD
For every JD keyword and responsibility, check whether my resume:
(a) covers it with strong evidence
(b) covers it weakly or only in the skills section
(c) does not cover it at all

Produce a gap table with three columns: [JD Requirement | Resume Coverage | Status: Strong / Weak / Missing]

---

### STEP 3 — Skills Section Cleanup
Audit every skill listed against two criteria:
(a) Is it evidenced in at least one bullet in Experience or Projects?
(b) Is it relevant to this specific JD?

Remove any skill that fails BOTH criteria. Flag skills that fail one. Do not remove skills that are JD keywords even if not in bullets.

---

### STEP 4 — Bullet Rewrites
For each Experience and Projects bullet, apply the following rules:

CONTENT rules:
- Every bullet must contain at least one JD keyword, bolded
- Every bullet must have a quantified metric (number, %, time saved, scale)
- No em-dashes. Use commas or restructure the sentence instead
- Maximum 200 characters per bullet (standard single-line resume length)
- No redundancy: if two bullets cover the same theme, merge or remove one

KEYWORD rules:
- Bold every JD keyword that appears in a bullet
- Do not bold generic words — only bold terms that directly match JD language
- Ensure these JD responsibility phrases appear bolded somewhere in bullets:
  [extract from Step 1 and insert here]

IMPACT rules:
- Lead with action verb
- Include human impact where possible (users impacted, time saved, team size)
- Frame work as production-scale where truthful ("designed for production deployment at scale")
- For ML roles: include model quality metrics (F1, precision, latency, accuracy, relevancy/faithfulness/coherence)

---

### STEP 5 — Gap Filling
For each MISSING item from Step 2, ask me:
"Do you have any experience with [X]? Even adjacent work counts."

Based on my answer, either:
- Write a new bullet if I have real experience
- Add it to the Skills section only if I have passing familiarity
- Skip it if I have nothing to back it up

For estimated/placeholder metrics (when I don't have exact numbers), use conservative, credible figures and flag them with [ESTIMATE — verify before submitting].

---

### STEP 6 — Redundancy Removal
After all rewrites, audit every bullet for:
- Overlap with another bullet (same theme, same role)
- Low signal relative to other bullets (vague, no metric, no JD keyword)
- Skills in the skills section that are no longer evidenced in any bullet after rewrites

Remove or merge anything that adds no unique signal.

---

### STEP 7 — Final Evaluation
Score the resume against the JD across these dimensions:

| Dimension | Score /20 | Notes |
|---|---|---|
| Minimum qualifications met | /20 | |
| Preferred qualifications met | /20 | |
| JD responsibilities covered | /20 | |
| Keyword density + bolding | /20 | |
| Bullet quality (metrics, length, impact) | /20 | |

**Total: /100**

List any remaining gaps and whether they are fixable without fabricating experience.

---

### FORMATTING RULES (apply throughout):
- Degrees: always write in full (Master of Science, Bachelor of Technology — never MS, BS, BTech)
- No em-dashes in bullet text
- Bullet length: max 200 characters
- Bold: JD keywords only, not generic phrases
- Skills: only include if evidenced in bullets OR directly in JD
- Estimated numbers: flag with [ESTIMATE]

---

### MY RESUME:
[paste resume text here]

### JOB DESCRIPTION:
[paste JD here]

### ADDITIONAL CONTEXT (optional):
- Real numbers I can confirm: [e.g. "TopGPT had 15 active users, saved ~40 min per session"]
- Leadership experience: [e.g. "Mentored 3-5 interns at TCS, led microservice API end-to-end"]
- Teaching/documentation: [e.g. "TA for 2 semesters, prepared Canvas materials for 100+ students"]
- Any other context the JD asks for that isn't on my resume: [e.g. "accessible technologies — I made course docs accessible on Canvas"]
```

---

## WHAT THIS PROMPT DOES

This prompt reproduces the full optimization pipeline applied to Sai Teja's resume for the Google SWE (AI/ML) role, including:

| Step | What it does |
|---|---|
| JD Analysis | Extracts every keyword, responsibility, and qualification |
| Resume Audit | Gaps table — strong / weak / missing per JD requirement |
| Skills Cleanup | Removes skills not evidenced in bullets and not in JD |
| Bullet Rewrites | Enforces JD keywords bolded, metrics, 200-char limit, no em-dashes |
| Gap Filling | Asks for real experience before adding anything; flags estimates |
| Redundancy Removal | Removes overlapping or low-signal bullets and stale skills |
| Final Evaluation | Scores resume /100 across 5 dimensions with remaining gap analysis |

## TIPS FOR BEST RESULTS

- **Paste the JD in full** — don't summarize it. The more verbatim JD language, the better keyword matching.
- **Fill in the Additional Context section** — real numbers and real experiences make the difference between a generic rewrite and a tailored one.
- **Run Step 5 interactively** — for gap-filling, answer each question honestly. The prompt will not fabricate; it only works with what you give it.
- **Verify all [ESTIMATE] flags** before submitting — conservative estimates are better than wrong ones.
- **Re-run Step 7** after any manual edits to check your score hasn't dropped.
