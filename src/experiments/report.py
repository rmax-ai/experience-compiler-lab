"""Iteration report (SPEC.md §17): static Markdown, no UI.

Renders an :class:`EvolutionResult` into the §17 block shape (training
success, observed patterns, candidate, validation delta, regressions,
decision) plus a provenance chain mapping failures -> evidence -> patch ->
eval -> decision for every proposed candidate. Deterministic: the output is
a pure function of the EvolutionResult — no timestamps, no IO beyond writing
the file.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from experiments.evolution import EvolutionResult

_TEMPLATE = Template("""\
# Iteration report — {{ result.workflow }}

Seed: {{ result.seed }}
Iterations run: {{ result.iterations | length }}
{% for it in result.iterations %}
## Iteration {{ it.iteration }}

Training: {{ it.runs_succeeded }} / {{ it.runs_created }} success
Observed patterns:
{% if it.new_record_ids -%}
{% for record_id in it.new_record_ids -%}
+ {{ record_id }}
{% endfor -%}
{% else -%}
(none)
{% endif -%}
{% if it.candidate_id %}
Candidate {{ it.candidate_id }}:
{{ it.lines_modified }} lines modified
Validation:
v{{ it.eval.from_version }} {{ "%.1f" | format(it.eval.previous_score * 100) }}% -> \
v{{ it.eval.to_version }} {{ "%.1f" | format(it.eval.candidate_score * 100) }}%
Regressions:
{{ it.eval.regressions }}
Decision:
{{ "ACCEPT" if it.decision == "accepted" else "REJECT" }}
{% else %}
Candidate: none (no patch proposed — loop stopped)
Decision: none
{% endif %}
{% endfor %}
## Provenance chain

{% if result.provenance -%}
{% for link in result.provenance -%}
- Iteration {{ link.iteration }}: failures [{{ link.failure_run_ids | join(', ') }}] \
-> evidence [{{ link.record_ids | join(', ') }}] \
-> patch {{ link.candidate_id }} \
-> eval {{ "%.2f" | format(link.previous_score) }} -> \
{{ "%.2f" | format(link.candidate_score) }} \
-> {{ "ACCEPTED" if link.decision == "accepted" else "REJECTED" }}
{% endfor -%}
{% else -%}
No candidates were proposed.
{% endif %}

---
The held-out test set was not used anywhere in this evolution run.
""")


def write_iteration_report(result: EvolutionResult, path: str) -> None:
    """Render ``result`` as a SPEC §17 Markdown report and write it to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_TEMPLATE.render(result=result), encoding="utf-8")
