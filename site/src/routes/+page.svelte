<script lang="ts">
	import { base } from '$app/paths';
	import { metrics, scenarios, stack, version } from '$lib/data/meta';

	const github = 'https://github.com/rmax-ai/experience-compiler-lab';
	const releases = 'https://github.com/rmax-ai/experience-compiler-lab/releases';
	const quickstart = `uv sync
source .venv/bin/activate
exp version
exp run train
exp mine
exp propose onboarding
exp eval candidate-01
exp promote candidate-01
exp evolve --iterations 10
exp compare
exp matrix --models fake --iterations 1
exp report`;

	const stores = [
		['experience/', 'Immutable observations', 'Append-only execution traces: the record of what happened.'],
		['knowledge/', 'Append-only interpretations', 'Evidence-backed knowledge records; supersede, never overwrite.'],
		['skills/', 'Mutable and reversible', 'Deployed procedural knowledge, changed only through provenance-tracked patches.']
	];

	const questions = [
		['H1', 'Does structured persistent knowledge produce better skill proposals than raw trajectory history?', 'Ablations compare raw recent traces with compiled knowledge.'],
		['H2', 'Do evolved skills improve unseen-task performance?', 'Candidate skills are assessed against a held-out validation set.'],
		['H3', 'Does keeping knowledge unavailable to the execution agent produce better reusable skills?', 'Execution and learning contexts remain separated.'],
		['H4', 'Do some skills transfer between models while others are model-specific compensation?', 'A cross-model transfer matrix executes every skill on every model.']
	];
</script>

<svelte:head>
	<title>Experience Compiler Lab</title>
	<meta name="description" content="Can agent experience be compiled into validated procedural knowledge?" />
</svelte:head>

<div class="min-h-screen overflow-x-hidden bg-slate-950 text-slate-300">
	<header class="border-b border-slate-800/90">
		<div class="mx-auto flex max-w-6xl items-center justify-between gap-5 px-6 py-5 lg:px-8">
			<a href="#top" class="text-sm font-semibold tracking-tight text-white sm:text-base">Experience Compiler Lab</a>
			<div class="flex items-center gap-3 text-xs sm:text-sm">
				<span class="rounded-full border border-indigo-400/30 bg-indigo-400/10 px-2.5 py-1 font-medium text-indigo-300">{version}</span>
				<a class="text-slate-400 transition hover:text-white" href={github}>github.com/rmax-ai/experience-compiler-lab</a>
			</div>
		</div>
	</header>

	<main id="top">
		<section class="relative border-b border-slate-900">
			<div class="absolute inset-0 -z-0 bg-[radial-gradient(circle_at_50%_0%,rgba(99,102,241,0.16),transparent_43%)]"></div>
			<div class="relative mx-auto max-w-6xl px-6 py-24 lg:px-8 lg:py-32">
				<p class="mb-5 font-mono text-sm font-medium text-indigo-400">A research harness for procedural learning</p>
				<h1 class="max-w-4xl text-4xl font-semibold tracking-tight text-white sm:text-6xl">Can agent experience be compiled into validated procedural knowledge?</h1>
				<p class="mt-7 max-w-3xl text-lg leading-8 text-slate-300">execution traces → evidence → persistent knowledge → candidate patch → independent evaluation → promotion / rejection.</p>
				<a class="mt-10 inline-flex rounded-lg bg-indigo-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400" href={github}>Explore on GitHub <span class="ml-2" aria-hidden="true">→</span></a>
			</div>
		</section>

		<section class="mx-auto max-w-6xl px-6 py-20 lg:px-8" aria-labelledby="pipeline-title">
			<div class="mb-9"><p class="eyebrow">The compiler loop</p><h2 id="pipeline-title">Failure is evidence—not a dead end.</h2></div>
			<img src={`${base}/pipeline-loop.svg`} alt="Pipeline loop from failure through evidence, hypothesis, patch, evaluation, and decision." class="w-full rounded-2xl border border-slate-800" />
		</section>

		<section class="border-y border-slate-800 bg-slate-900/20" aria-labelledby="stores-title">
			<div class="mx-auto max-w-6xl px-6 py-20 lg:px-8"><p class="eyebrow">Three stores, one invariant</p><h2 id="stores-title">Keep observations, interpretations, and procedures distinct.</h2>
				<div class="mt-10 grid gap-5 md:grid-cols-3">{#each stores as store}<article class="rounded-xl border border-slate-800 bg-slate-900/50 p-6"><code class="text-base font-semibold text-indigo-400">{store[0]}</code><h3 class="mt-6 text-lg font-medium text-white">{store[1]}</h3><p class="mt-3 leading-7 text-slate-400">{store[2]}</p></article>{/each}</div>
			</div>
		</section>

		<section class="mx-auto max-w-6xl px-6 py-20 lg:px-8" aria-labelledby="questions-title">
			<p class="eyebrow">Research questions</p><h2 id="questions-title">Four claims, each with an experimental mechanism.</h2>
			<div class="mt-10 grid gap-5 md:grid-cols-2">{#each questions as question}<article class="rounded-xl border border-slate-800 bg-slate-900/50 p-6"><span class="font-mono text-sm font-semibold text-indigo-400">{question[0]}</span><h3 class="mt-3 text-lg font-medium leading-7 text-white">{question[1]}</h3><p class="mt-4 border-l-2 border-indigo-400/50 pl-4 leading-7 text-slate-400">{question[2]}</p></article>{/each}</div>
		</section>

		<section class="border-y border-slate-800 bg-slate-900/20" aria-labelledby="harness-title">
			<div class="mx-auto max-w-6xl px-6 py-20 lg:px-8"><p class="eyebrow">How the harness works</p><h2 id="harness-title">A deterministic enterprise onboarding world.</h2>
				<div class="mt-8 grid gap-8 lg:grid-cols-[1.25fr_.75fr]"><p class="text-lg leading-8 text-slate-300">A {scenarios.total}-scenario onboarding world: {scenarios.train} train / {scenarios.validation} validation / {scenarios.heldOut} held-out. It exposes {scenarios.tools} tools—<span class="font-mono text-sm text-slate-300">get_employee, get_policy, get_inventory, assign_device, create_procurement_request, grant_access, create_ticket, complete_onboarding</span>—with deterministic final-state graders and fixed seeds. Every run captures token and cost metrics as evidence.</p><div class="rounded-xl border border-slate-800 bg-black p-6"><p class="font-mono text-xs uppercase tracking-widest text-slate-500">Current status</p><p class="mt-3 text-lg font-medium text-white">v0.1.0 released</p><p class="mt-3 leading-7 text-slate-400">All milestones M0–M6 shipped. Real-model (live LLM) experiments are next; the harness currently runs deterministically with a scripted fake model—no API key needed, no cost.</p></div></div>
				<div class="mt-10 flex flex-wrap gap-2">{#each stack as item}<span class="rounded-full border border-slate-800 px-3 py-1.5 text-sm text-slate-400">{item}</span>{/each}</div>
			</div>
		</section>

		<section class="mx-auto max-w-6xl px-6 py-20 lg:px-8" aria-labelledby="studies-title">
			<p class="eyebrow">M5 + M6</p><h2 id="studies-title">Ablation and transfer studies.</h2>
			<div class="mt-10 grid gap-6 lg:grid-cols-2"><article class="rounded-xl border border-slate-800 bg-slate-900/50 p-6"><h3 class="text-xl font-medium text-white">Ablation study</h3><p class="mt-3 leading-7 text-slate-400">Four configurations, with the held-out set never touched during training.</p><div class="mt-6 overflow-hidden rounded-lg border border-slate-800"><table class="w-full text-left text-sm"><thead class="bg-black text-slate-400"><tr><th class="px-4 py-3 font-medium">Configuration</th><th class="px-4 py-3 font-medium">Context</th></tr></thead><tbody class="divide-y divide-slate-800 text-slate-300"><tr><td class="px-4 py-3">baseline</td><td class="px-4 py-3">no persistence</td></tr><tr><td class="px-4 py-3">trace2skill</td><td class="px-4 py-3">raw recent traces only</td></tr><tr><td class="px-4 py-3">memory</td><td class="px-4 py-3">knowledge in executor context</td></tr><tr><td class="px-4 py-3">compiler</td><td class="px-4 py-3">full pipeline</td></tr></tbody></table></div></article><article class="rounded-xl border border-slate-800 bg-slate-900/50 p-6"><h3 class="text-xl font-medium text-white">Cross-model transfer matrix</h3><p class="mt-3 leading-7 text-slate-400">Train a skill with each model, then execute every skill on every model. The skill-source × executor matrix tests whether skills encode environment knowledge or model compensations.</p><div class="mt-8 rounded-lg border border-indigo-400/20 bg-indigo-400/5 p-4"><p class="font-mono text-sm text-indigo-300">results/transfer-matrix.csv</p><p class="mt-2 text-sm leading-6 text-slate-400">The M6 output path for the transfer matrix.</p></div></article></div>
		</section>

		<section class="border-y border-slate-800 bg-slate-900/20" aria-labelledby="quickstart-title"><div class="mx-auto max-w-6xl px-6 py-20 lg:px-8"><p class="eyebrow">Quickstart</p><h2 id="quickstart-title">Run the loop locally.</h2><pre class="mt-8 overflow-x-auto rounded-xl border border-slate-800 bg-black p-6 text-sm leading-7 text-slate-300"><code>{quickstart}</code></pre></div></section>

		<section class="mx-auto max-w-6xl px-6 py-10 lg:px-8" aria-label="Project metrics"><div class="grid gap-px overflow-hidden rounded-xl border border-slate-800 bg-slate-800 sm:grid-cols-5">{#each metrics as metric}<div class="bg-slate-950 px-5 py-5 text-center text-sm font-medium text-slate-300">{metric}</div>{/each}</div></section>
	</main>

	<footer class="border-t border-slate-800"><div class="mx-auto flex max-w-6xl flex-col gap-3 px-6 py-8 text-sm text-slate-600 sm:flex-row sm:items-center sm:justify-between lg:px-8"><span>MIT</span><div class="flex gap-5"><a class="transition hover:text-slate-300" href={github}>GitHub</a><a class="transition hover:text-slate-300" href={releases}>Releases</a></div></div></footer>
</div>

<style>
	:global(.eyebrow) { margin-bottom: 0.75rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.75rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: #818cf8; }
	:global(h2) { max-width: 44rem; font-size: clamp(1.875rem, 4vw, 2.5rem); font-weight: 600; letter-spacing: -.025em; color: white; }
</style>
