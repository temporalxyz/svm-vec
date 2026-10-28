# svm-vec

## documentation discipline

Only necessary documentation in code. Before adding a comment or doc, ask: do we really need this HERE, NOW? If not, leave it out.

- Docs state exactly what the code does and why it is done this way. Plain, concise, human. No AI-sounding filler, no restating the code, no narration.
- Notes that are uncertain but might help another agent go in `slop/`, not in the code. Nothing in `slop/` is authoritative or load-bearing.

## repo rules

- `SPEC.md` is the contract between `sbpf/`, `sbpf-linker/` and `svm-intrinsics/`. Change an op in all three plus the runtime tests, or not at all.
- `sbpf/` and `sbpf-linker/` are forks meant to go upstream. Keep their diffs minimal and in upstream style, put new code in new modules, and add nothing repo-specific (no CLAUDE.md, no slop) inside them.
- Programs without vector ops must compile and run exactly as before. Keep the test for that passing.
- Benchmark numbers in `results/` are measured, never estimated. Mark anything estimated, or put it in `slop/`.
- Commits are signed with a hardware key: don't commit or push unless asked. No Claude or co-author trailers.
