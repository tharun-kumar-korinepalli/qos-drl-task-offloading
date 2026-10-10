# Reviewer-round progress (v3)

Status file so the work can resume after a break. Updated 2026-10-09.

## Done
- docs/self_review.md: 10 major and 5 minor reviewer objections, each with the planned answer.
- experiments/reward_analysis.py -> results/summary/reward_analysis.json. Proposition: the +1 bonus adds 1 to every penalty (10:2 acts as 11:3). Corrected penalties (10.67, 1.33) give the exact optimum 0.2614. Flat gap: 17% (ratio 2), 95% (5), 590% (50).
- qos_offloading/general.py: N heterogeneous edge nodes, K classes, random background, cloud that meets a limit with probability pi_c, sparse exact DP, exact evaluation, rules (JSQ, SLA rule, reservation, reservation with node sets). tests/test_general.py passes (10 tests).
- experiments/scenario_reference.py -> results/summary/scenario_reference.json (rules, optimum, DP policy of each reward, gamma 0.99, frontier).
- experiments/dp_scaling.py -> results/summary/dp_scaling.json (N = 2..5).
- Tuning (72 runs) done, results/summary/tuning_selection.json. Same setting chosen for both rewards for every algorithm.
- Paper draft: /home/claude/journal_paper_v3 (part_front, part_model, part_exact, part_method; gen/fill.py fills tables from the JSON files; gen/make_architecture.py).

## Status
All stages finished on 2026-10-10: tuning (72), main (70), larger systems with the nominal setting (90), per-system tuning (36), larger systems tuned per system (90), changes of the system (100), published designs (30). Analysis in results/summary/summary_v3.json, tables in RESULTS_v3.md, paper in paper/.
